from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from enum import Enum

import cv2

try:
    from tetris.HandGestures import (
        ACTION_HARD_DROP,
        ACTION_HOLD,
        ACTION_LEFT,
        ACTION_RETRY,
        ACTION_RIGHT,
        ACTION_ROTATE,
        ACTION_SOFT_DROP,
        ALL_ACTIONS,
        GestureDetector,
        GestureState,
    )
    from tetris.HandTrackingModule import HandTracker
except ModuleNotFoundError:
    from HandGestures import (
        ACTION_HARD_DROP,
        ACTION_HOLD,
        ACTION_LEFT,
        ACTION_RETRY,
        ACTION_RIGHT,
        ACTION_ROTATE,
        ACTION_SOFT_DROP,
        ALL_ACTIONS,
        GestureDetector,
        GestureState,
    )
    from HandTrackingModule import HandTracker


# ──────────────────────────────────────────────────────────────────────
# Gesture → keyboard binding (replaces the old comment block).
# Each action can fire one OR MORE keys; all listed keys are
# pressed/released together so a game can be played with either the
# arrow keys (default) or the numpad.
# ──────────────────────────────────────────────────────────────────────
class KeyboardKey(Enum):
    """Windows virtual-key codes used by the gesture→keyboard mapper."""

    LEFT = 0x25      # VK_LEFT
    UP = 0x26        # VK_UP
    RIGHT = 0x27     # VK_RIGHT
    DOWN = 0x28      # VK_DOWN
    SPACE = 0x20     # VK_SPACE
    NUMPAD2 = 0x62   # VK_NUMPAD2
    NUMPAD4 = 0x64   # VK_NUMPAD4
    NUMPAD6 = 0x66   # VK_NUMPAD6
    NUMPAD8 = 0x68   # VK_NUMPAD8
    NUMPAD9 = 0x69   # VK_NUMPAD9
    R = 0x52         # VK_R  (used for ACTION_RETRY)
    C = 0x43         # VK_C
    SHIFT = 0x10     # VK_SHIFT

# Action → list of keys to press when the action becomes active.
# The two-key per action matches the original comment table exactly.
ACTION_KEY_MAP: dict[str, tuple[KeyboardKey, ...]] = {
    ACTION_LEFT:       (KeyboardKey.LEFT,  KeyboardKey.NUMPAD4),
    ACTION_RIGHT:      (KeyboardKey.RIGHT, KeyboardKey.NUMPAD6),
    ACTION_ROTATE:     (KeyboardKey.UP,    KeyboardKey.NUMPAD9),
    ACTION_HARD_DROP:  (KeyboardKey.SPACE, KeyboardKey.NUMPAD8),
    ACTION_SOFT_DROP:  (KeyboardKey.DOWN,  KeyboardKey.NUMPAD2),
    ACTION_RETRY:      (KeyboardKey.R,),
    ACTION_HOLD:       (KeyboardKey.SHIFT, KeyboardKey.C),
}


# ──────────────────────────────────────────────────────────────────────
# Win32 SendInput wrapper — no external dependency, talks to user32
# directly to synthesize real OS key events.
# ──────────────────────────────────────────────────────────────────────
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.LPARAM),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.LPARAM),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


# Global hotkey constants (Win32 RegisterHotKey).
# Only the modifiers actually used by the toggle hotkey are defined.
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_NOREPEAT = 0x4000
# Toggle hotkey: Ctrl+Alt+G  (G = "gestures")
_TOGGLE_HOTKEY_ID = 1
_TOGGLE_HOTKEY_MODS = _MOD_CONTROL | _MOD_ALT | _MOD_NOREPEAT
_TOGGLE_VK = 0x47  # 'G'
_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001


class Win32Keyboard:
    """Send real keystrokes via the Win32 SendInput API.

    Holds a suspend flag (a `threading.Event`) so a global hotkey can halt
    synthesis without destroying the object. All methods are no-ops on
    non-Windows platforms so the module remains importable cross-platform.

    A *Protocol*-style duck-typed test double only needs to implement
    `press(key) -> bool` and `release(key) -> bool` (see
    `GestureKeyboardDispatcher` for the consumer).
    """

    def __init__(self) -> None:
        self._user32: ctypes.WinDLL | None = None
        self._kernel32: ctypes.WinDLL | None = None
        if hasattr(ctypes, "WinDLL"):
            try:
                self._user32 = ctypes.WinDLL("user32", use_last_error=True)
                self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            except OSError:
                self._user32 = None
                self._kernel32 = None

        # threading.Event is atomic and has well-defined memory ordering;
        # is_set() = "synthesis is currently halted".
        self._suspended = threading.Event()
        self._hotkey_thread: threading.Thread | None = None
        self._hotkey_stop = threading.Event()
        self._shutdown_called = False

        if self._user32 is not None and sys.platform == "win32":
            self._start_hotkey_listener()

    # ── suspension API ─────────────────────────────────────────────
    @property
    def suspended(self) -> bool:
        """True iff keyboard synthesis is currently halted."""
        return self._suspended.is_set()

    def suspend(self) -> None:
        """Halt keyboard synthesis. Idempotent. Thread-safe."""
        self._suspended.set()

    def resume(self) -> None:
        """Resume keyboard synthesis. Idempotent. Thread-safe."""
        self._suspended.clear()

    def toggle(self) -> bool:
        """Toggle keyboard synthesis. Returns the new state (True = suspended). Thread-safe."""
        if self._suspended.is_set():
            self._suspended.clear()
            return False
        self._suspended.set()
        return True

    # ── low-level send ─────────────────────────────────────────────
    def _send(self, vk: int, key_up: bool) -> bool:
        """Send one synthetic key event. Returns True on success.

        Returns False (without raising) if:
          - the backend is not available (non-Windows)
          - synthesis is currently suspended
          - user32.SendInput reports 0 events delivered
        """
        if self._user32 is None or self._suspended.is_set():
            return False
        inp = _INPUT()
        inp.type = _INPUT_KEYBOARD
        inp.u.ki.wVk = vk
        inp.u.ki.wScan = 0
        inp.u.ki.dwFlags = _KEYEVENTF_KEYUP if key_up else 0
        inp.u.ki.time = 0
        inp.u.ki.dwExtraInfo = 0
        sent = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if sent == 0 and self._kernel32 is not None:
            err = ctypes.get_last_error()
            print(
                f"[Win32Keyboard] SendInput failed for vk=0x{vk:02X} "
                f"key_up={key_up} (Win32 error {err})"
            )
        return sent != 0

    def press(self, key: KeyboardKey) -> bool:
        """Simulate a key-down event. Returns True on success."""
        return self._send(int(key.value), key_up=False)

    def release(self, key: KeyboardKey) -> bool:
        """Simulate a key-up event. Returns True on success."""
        return self._send(int(key.value), key_up=True)

    # ── global hotkey listener ─────────────────────────────────────
    def _start_hotkey_listener(self) -> None:
        """Spawn a daemon thread that registers + listens for the toggle hotkey.

        IMPORTANT: Win32 hotkeys are thread-scoped — `RegisterHotKey(NULL, ...)`
        posts `WM_HOTKEY` to the *calling* thread's message queue, and only
        that thread can `UnregisterHotKey` it. Both calls therefore happen
        inside the listener thread body, paired in a try/finally.
        """
        def _pump() -> None:
            # Register from THIS thread so the WM_HOTKEY lands in THIS
            # thread's message queue.
            if not self._user32.RegisterHotKey(
                None, _TOGGLE_HOTKEY_ID, _TOGGLE_HOTKEY_MODS, _TOGGLE_VK
            ):
                err = ctypes.get_last_error()
                print(
                    f"[Win32Keyboard] RegisterHotKey failed (Win32 error "
                    f"{err}). Ctrl+Alt+G toggle will not work; use "
                    "suspend()/resume() instead."
                )
                return
            try:
                msg = wintypes.MSG()
                while not self._hotkey_stop.is_set():
                    if self._user32.PeekMessageW(
                        ctypes.byref(msg), None, _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE
                    ):
                        if msg.wParam == _TOGGLE_HOTKEY_ID:
                            now_suspended = self.toggle()
                            print(
                                f"[Win32Keyboard] Keyboard synthesis "
                                f"{'SUSPENDED' if now_suspended else 'RESUMED'} "
                                "(Ctrl+Alt+G)"
                            )
                    else:
                        # Avoid pegging a CPU core.
                        self._hotkey_stop.wait(0.05)
            finally:
                # Unregister from the same thread that registered.
                self._user32.UnregisterHotKey(None, _TOGGLE_HOTKEY_ID)

        self._hotkey_thread = threading.Thread(
            target=_pump, name="GestureHotkeyListener", daemon=True
        )
        self._hotkey_thread.start()

    def shutdown(self) -> None:
        """Stop the hotkey listener and release the registered hotkey.

        Idempotent: safe to call multiple times.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._hotkey_stop.set()
        if self._hotkey_thread is not None:
            self._hotkey_thread.join(timeout=0.5)

    def __del__(self) -> None:
        # Safety net for callers that forget shutdown() (e.g. short-lived
        # test scripts). Best-effort: never raise during interpreter
        # teardown.
        try:
            self.shutdown()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────
# GestureState → OS key events.
# A key is pressed on a rising edge (action becomes True) and released
# on a falling edge (action becomes False).  This way, holding a gesture
# for many frames still results in exactly one keypress per gesture.
# ──────────────────────────────────────────────────────────────────────
class GestureKeyboardDispatcher:
    """Bind a `GestureState` action dict to real keystrokes via a keyboard backend.

    The keyboard backend must expose `press(key) -> bool` and
    `release(key) -> bool`; `Win32Keyboard` satisfies this, as does any
    test double.

    Only action names listed in `ALL_ACTIONS` are tracked. Unknown
    action names in `state.actions` are silently ignored to keep
    `_prev` bounded.
    """

    def __init__(
        self,
        keyboard: Win32Keyboard,
        action_key_map: dict[str, tuple[KeyboardKey, ...]] | None = None,
    ) -> None:
        self._keyboard = keyboard
        # Explicit `is None` check: an empty dict means "no mappings",
        # not "fall back to defaults".
        self._action_key_map: dict[str, tuple[KeyboardKey, ...]] = (
            action_key_map if action_key_map is not None else ACTION_KEY_MAP
        )
        self._known: frozenset[str] = frozenset(ALL_ACTIONS)
        self._prev: dict[str, bool] = {name: False for name in ALL_ACTIONS}

    def dispatch(self, state: GestureState) -> None:
        """Update held keys to match the current action set.

        Single pass over `self._known` (the canonical action set) that
        handles rising edges and falling edges. Unknown keys present
        in `state.actions` are silently dropped. `self._prev[name]`
        is only updated when the OS operation succeeded, so a transient
        `SendInput` failure causes a retry on the next frame instead
        of diverging from the OS.
        """
        current = state.actions

        # Only iterate over known action names so an unexpected key in
        # `state.actions` cannot grow `_prev` without bound. Any held
        # key for a now-unknown action is still released below.
        for action_name in self._known:
            was_active = self._prev.get(action_name, False)
            is_active = current.get(action_name, False)
            if is_active and not was_active:
                if self._press_action(action_name):
                    self._prev[action_name] = True
                # else: leave was_active=False; will retry next frame.
            elif was_active and not is_active:
                if self._release_action(action_name):
                    self._prev[action_name] = False
                # else: leave was_active=True; will retry next frame.
            # else: steady state — `_prev` already equals the desired
            # value, no work to do.

    def release_all(self) -> None:
        """Best-effort release of every key currently held by the dispatcher.

        Called from `main()`'s exit path. The return value of each
        underlying release is intentionally discarded, and `_prev` is
        unconditionally cleared, because the process is about to
        terminate and there is no caller that could act on a failure.
        """
        for name in self._known:
            was_active = self._prev.get(name, False)
            if was_active and self._action_key_map.get(name, ()):
                self._release_action(name)
            self._prev[name] = False

    def pressed_actions(self) -> list[str]:
        """Return the action names currently considered 'pressed'.

        The list reflects the dispatcher's `_prev` state, which is only
        set to `True` after a successful `press` and only cleared after
        a successful `release`. Order matches `ALL_ACTIONS`.
        """
        return [name for name in ALL_ACTIONS if self._prev.get(name, False)]

    def pressed_keys_for(self, action_name: str) -> tuple[KeyboardKey, ...]:
        """Return the key tuple mapped to `action_name` (empty if unmapped)."""
        return self._action_key_map.get(action_name, ())

    def _press_action(self, action_name: str) -> bool:
        """Press all keys for `action_name`. True iff every key was sent.

        On partial failure (one key's `press` succeeded, a later key's
        failed) the keys that *did* succeed are released before
        returning, so the OS state matches the dispatcher's belief and
        a self-heal on the next frame starts from a clean slate.
        """
        keys = self._action_key_map.get(action_name, ())
        if not keys:
            return True
        rolled_back: list[KeyboardKey] = []
        all_ok = True
        for key in keys:
            if self._keyboard.press(key):
                rolled_back.append(key)
            else:
                all_ok = False
        if not all_ok:
            for key in rolled_back:
                self._keyboard.release(key)
        return all_ok

    def _release_action(self, action_name: str) -> bool:
        """Release all keys for `action_name`. True iff every key was sent.

        No useful rollback exists for a partial release failure — a
        physical key cannot be "un-released" — so the call simply
        attempts all keys and reports the AND.
        """
        keys = self._action_key_map.get(action_name, ())
        if not keys:
            return True
        return all(self._keyboard.release(key) for key in keys)
    
    @staticmethod
    def draw_pressed_keys_overlay( img, dispatcher: "GestureKeyboardDispatcher", keyboard: "Win32Keyboard",) -> None:
        draw_pressed_keys_overlay(img, dispatcher, keyboard)


# ──────────────────────────────────────────────────────────────────────
# Camera loop
# ──────────────────────────────────────────────────────────────────────
_PRETTY_KEY_NAMES: dict[KeyboardKey, str] = {
    KeyboardKey.LEFT: "Left",
    KeyboardKey.UP: "Up",
    KeyboardKey.RIGHT: "Right",
    KeyboardKey.DOWN: "Down",
    KeyboardKey.SPACE: "Space",
    KeyboardKey.R: "R",
    KeyboardKey.NUMPAD2: "Numpad 2",
    KeyboardKey.NUMPAD4: "Numpad 4",
    KeyboardKey.NUMPAD6: "Numpad 6",
    KeyboardKey.NUMPAD8: "Numpad 8",
    KeyboardKey.NUMPAD9: "Numpad 9",
}


def draw_pressed_keys_overlay(
    img,
    dispatcher: "GestureKeyboardDispatcher",
    keyboard: "Win32Keyboard",
) -> None:
    """Render a top-left text overlay showing the currently pressed keys.

    Layout:
      Line 1: status — "SUSPENDED (Ctrl+Alt+G to resume)" in red, or
              "ACTIVE" in green.
      Line 2+: one line per pressed action, formatted as
               "  <action>: <Key1>, <Key2>".
    If no actions are pressed (and not suspended), shows "Held: (none)".
    """
    pad_x = 12
    line1_y = 80
    line_step = 24

    if keyboard.suspended:
        cv2.putText(
            img,
            "SUSPENDED (Ctrl+Alt+G to resume)",
            (pad_x, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    cv2.putText(
        img,
        "ACTIVE",
        (pad_x, line1_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    active = dispatcher.pressed_actions()
    if not active:
        cv2.putText(
            img,
            "Held: (none)",
            (pad_x, line1_y + line_step),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return

    for i, action_name in enumerate(active):
        keys = dispatcher.pressed_keys_for(action_name)
        key_names = ", ".join(_PRETTY_KEY_NAMES.get(k, k.name) for k in keys)
        cv2.putText(
            img,
            f"  {action_name}: {key_names}",
            (pad_x, line1_y + (i + 1) * line_step),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main(camera_index: int = 0) -> None:
    # All resources (camera, keyboard, dispatcher) are acquired inside
    # this try-block so a failure to construct any of them still cleans
    # up the ones already created.
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return

        print(
            "Gesture keyboard synthesis is ACTIVE. Press Ctrl+Alt+G to "
            "toggle. Focus must be on the target app (e.g. your game)."
        )

        keyboard = Win32Keyboard()
        dispatcher = GestureKeyboardDispatcher(keyboard)

        try:
            with HandTracker(max_hands=2) as tracker:
                with GestureDetector() as detector:
                    while True:
                        success, img = cap.read()
                        if not success:
                            break

                        img, hands = tracker.find_hands(img, draw=True)
                        img = tracker.label_hands(img)
                        state = detector.update(hands)
                        detector.draw_gesture_overlay(img, state)
                        dispatcher.dispatch(state)
                        draw_pressed_keys_overlay(img, dispatcher, keyboard)

                        cv2.imshow("Hand Gestures", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            dispatcher.release_all()
            keyboard.shutdown()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
