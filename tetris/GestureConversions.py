from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes
from enum import Enum

import cv2

from tetris.HandGestures import (
    ACTION_HARD_DROP,
    ACTION_HOLD,
    ACTION_LEFT,
    ACTION_RETRY,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_ROTATE_CCW,
    ACTION_SOFT_DROP,
    ALL_ACTIONS,
    GestureDetector,
    GestureState,
)
from tetris.HandTrackingModule import HandTracker


# An action can map to multiple keys that fire together, so the game
# sees either arrow keys or the numpad, not just one.
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
    CONTROL = 0x11   # VK_CONTROL
    Z = 0x5A         # VK_Z

# Action → keys pressed/released when the action is active.
ACTION_KEY_MAP: dict[str, tuple[KeyboardKey, ...]] = {
    ACTION_LEFT:       (KeyboardKey.LEFT,  KeyboardKey.NUMPAD4),
    ACTION_RIGHT:      (KeyboardKey.RIGHT, KeyboardKey.NUMPAD6),
    ACTION_ROTATE:     (KeyboardKey.UP,    KeyboardKey.NUMPAD9),
    ACTION_ROTATE_CCW: (KeyboardKey.CONTROL, KeyboardKey.Z),
    ACTION_HARD_DROP:  (KeyboardKey.SPACE, KeyboardKey.NUMPAD8),
    ACTION_SOFT_DROP:  (KeyboardKey.DOWN,  KeyboardKey.NUMPAD2),
    ACTION_RETRY:      (KeyboardKey.R,),
    ACTION_HOLD:       (KeyboardKey.SHIFT, KeyboardKey.C),
}


# Win32 SendInput wrapper — synthesizes real OS key events via user32.
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


# Win32 RegisterHotKey constants for the toggle hotkey.
_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_NOREPEAT = 0x4000
_TOGGLE_HOTKEY_ID = 1
_TOGGLE_HOTKEY_MODS = _MOD_CONTROL | _MOD_ALT | _MOD_NOREPEAT
_TOGGLE_VK = 0x47  # 'G'
_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001


class Win32Keyboard:
    """Send keystrokes via the Win32 SendInput API.

    A `threading.Event` gates synthesis so the hotkey can halt it
    without destroying the object. No-ops on non-Windows. Test doubles
    only need `press(key) -> bool` and `release(key) -> bool`.
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

        # is_set() means synthesis is halted.
        self._suspended = threading.Event()
        self._hotkey_thread: threading.Thread | None = None
        self._hotkey_stop = threading.Event()
        self._shutdown_called = False
        self._toggle_lock = threading.Lock()
        self._last_error_time = 0.0
        self._inp = _INPUT()  # Pre-allocated, reused by _send()

        if self._user32 is not None and sys.platform == "win32":
            self._start_hotkey_listener()

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
        with self._toggle_lock:
            if self._suspended.is_set():
                self._suspended.clear()
                return False
            self._suspended.set()
            return True

    def _send(self, vk: int, key_up: bool) -> bool:
        """Send one synthetic key event. Returns False (no raise) if the
        backend is missing, synthesis is suspended, or SendInput delivered 0.
        """
        if self._user32 is None or self._suspended.is_set():
            return False
        inp = self._inp
        inp.type = _INPUT_KEYBOARD
        inp.u.ki.wVk = vk
        inp.u.ki.wScan = 0
        inp.u.ki.dwFlags = _KEYEVENTF_KEYUP if key_up else 0
        inp.u.ki.time = 0
        inp.u.ki.dwExtraInfo = 0
        sent = self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        if sent == 0 and self._kernel32 is not None:
            now = time.monotonic()
            if now - self._last_error_time > 1.0:
                err = ctypes.get_last_error()
                print(
                    f"[Win32Keyboard] SendInput failed for vk=0x{vk:02X} "
                    f"key_up={key_up} (Win32 error {err}). Further errors "
                    "will be rate-limited."
                )
                self._last_error_time = now
        return sent != 0

    def press(self, key: KeyboardKey) -> bool:
        """Simulate a key-down event. Returns True on success."""
        return self._send(int(key.value), key_up=False)

    def release(self, key: KeyboardKey) -> bool:
        """Simulate a key-up event. Returns True on success."""
        return self._send(int(key.value), key_up=True)

    def _start_hotkey_listener(self) -> None:
        """Spawn a daemon thread that registers + listens for the toggle hotkey.

        Hotkeys are thread-scoped: RegisterHotKey posts WM_HOTKEY to the
        caller's queue, and only that thread can UnregisterHotKey. So
        both calls live in the listener body, paired in try/finally.
        """
        def _pump() -> None:
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
        """Stop the hotkey listener and release the registered hotkey. Idempotent."""
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._hotkey_stop.set()
        if self._hotkey_thread is not None:
            self._hotkey_thread.join(timeout=0.5)


class GestureKeyboardDispatcher:
    """Bind a `GestureState` action dict to real keystrokes.

    The keyboard backend needs `press(key) -> bool` and `release(key) -> bool`.
    Unknown action names in `state.actions` are ignored.
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

        `_prev` is only updated when the OS call succeeded, so a
        transient SendInput failure retries next frame instead of
        diverging from the OS.
        """
        current = state.actions

        # Only iterate known names so `state.actions` can't grow `_prev`.
        for action_name in self._known:
            was_active = self._prev.get(action_name, False)
            is_active = current.get(action_name, False)
            if is_active and not was_active:
                if self._press_action(action_name):
                    self._prev[action_name] = True
            elif was_active and not is_active:
                if self._release_action(action_name):
                    self._prev[action_name] = False

    def release_all(self) -> None:
        """Best-effort release of every held key. Exit-path only."""
        for name in self._known:
            was_active = self._prev.get(name, False)
            if was_active and self._action_key_map.get(name, ()):
                self._release_action(name)
            self._prev[name] = False

    def pressed_actions(self) -> list[str]:
        """Action names currently considered 'pressed' (in ALL_ACTIONS order)."""
        return [name for name in ALL_ACTIONS if self._prev.get(name, False)]

    def pressed_keys_for(self, action_name: str) -> tuple[KeyboardKey, ...]:
        """Return the key tuple mapped to `action_name` (empty if unmapped)."""
        return self._action_key_map.get(action_name, ())

    def _press_action(self, action_name: str) -> bool:
        """Press all keys for `action_name`. Rolls back partial successes."""
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
        """Release all keys for `action_name`. No useful rollback for partial failure."""
        keys = self._action_key_map.get(action_name, ())
        if not keys:
            return True
        results = [self._keyboard.release(key) for key in keys]
        return all(results)
    
    @staticmethod
    def draw_pressed_keys_overlay( img, dispatcher: "GestureKeyboardDispatcher", keyboard: "Win32Keyboard",) -> None:
        draw_pressed_keys_overlay(img, dispatcher, keyboard)


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
    KeyboardKey.CONTROL: "Control",
    KeyboardKey.Z: "Z",
}


def draw_pressed_keys_overlay(
    img,
    dispatcher: "GestureKeyboardDispatcher",
    keyboard: "Win32Keyboard",
) -> None:
    """Top-left overlay: status line, then one line per pressed action."""
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


