from __future__ import annotations

from enum import Enum
import cv2

try:
    from tetris.GestureConversions import (
        ACTION_KEY_MAP,
        GestureKeyboardDispatcher,
        KeyboardKey,
        Win32Keyboard,
    )
    from tetris.HandGestures import (
        ACTION_HARD_DROP,
        ACTION_HOLD,
        ACTION_LEFT,
        ACTION_RETRY,
        ACTION_RIGHT,
        ACTION_ROTATE,
        ACTION_SOFT_DROP,
        GestureState,
        GestureDetector,
    )
    from tetris.HandTrackingModule import HandTracker
except ModuleNotFoundError:
    from GestureConversions import (
        ACTION_KEY_MAP,
        GestureKeyboardDispatcher,
        KeyboardKey,
        Win32Keyboard,
    )
    from HandGestures import (
        ACTION_HARD_DROP,
        ACTION_HOLD,
        ACTION_LEFT,
        ACTION_RETRY,
        ACTION_RIGHT,
        ACTION_ROTATE,
        ACTION_SOFT_DROP,
        GestureState,
        GestureDetector,
    )
    from HandTrackingModule import HandTracker

class ActionMode(Enum):
    CONTINUOUS = "continuous"
    SINGLE = "single"


TETRIS_ACTION_MODES: dict[str, ActionMode] = {
    ACTION_LEFT: ActionMode.CONTINUOUS,
    ACTION_RIGHT: ActionMode.CONTINUOUS,
    ACTION_SOFT_DROP: ActionMode.CONTINUOUS,
    ACTION_ROTATE: ActionMode.SINGLE,
    ACTION_HARD_DROP: ActionMode.SINGLE,
    ACTION_RETRY: ActionMode.CONTINUOUS,
    ACTION_HOLD: ActionMode.SINGLE,
}


class TetrisKeyboardDispatcher(GestureKeyboardDispatcher):
    """GestureKeyboardDispatcher variant with per-action mode + falling-edge grace.  """
    def __init__(
        self,
        keyboard: Win32Keyboard,
        action_key_map: dict[str, tuple[KeyboardKey, ...]] = ACTION_KEY_MAP,
        action_modes: dict[str, ActionMode] = TETRIS_ACTION_MODES,
        grace_frames: int = 3,
    ) -> None:
        super().__init__(keyboard, action_key_map=action_key_map)
        self._modes: dict[str, ActionMode] = dict(action_modes)
        if grace_frames < 0:
            raise ValueError("grace_frames must be >= 0")
        self._grace_frames: int = grace_frames
        self._grace: dict[str, int] = {name: 0 for name in self._known}

    def dispatch(self, state: GestureState) -> None:
        """Update held or sent keys to match the current action set.

        CONTINUOUS actions use the base dispatcher's press-on-rising,
        release-on-falling logic, so the OS sees a held key and its
        built-in autorepeat produces repeated events.

        SINGLE actions fire a press+release on the rising edge and
        then become inert for the rest of the gesture. The falling
        edge clears the internal latch so the next rising edge fires
        a fresh tap. This avoids the autorepeat that a held key
        would produce, which is wrong for discrete actions like
        rotate and hard-drop.

        Both modes defer the actual release / latch-clear by up to
        `grace_frames` frames. Any active frame resets the grace
        counter to zero, so a single transient false negative in
        the middle of a continuous slide does not interrupt the
        OS's autorepeat or trigger a spurious re-fire. The
        deferred release fires on the first frame after the grace
        counter has been decremented to zero AND the input is
        still inactive.

        Partial-failure handling matches the base dispatcher: an
        OS call that returns False leaves `self._prev[action_name]`
        at its prior value, so the next frame retries the
        transition instead of silently diverging from the OS.
        """
        current = state.actions
        for action_name in self._known:
            was_active = self._prev.get(action_name, False)
            is_active = current.get(action_name, False)
            mode = self._modes.get(action_name, ActionMode.CONTINUOUS)
            if is_active:
                self._grace[action_name] = 0
                if not was_active:
                    if mode is ActionMode.CONTINUOUS:
                        if self._press_action(action_name):
                            self._prev[action_name] = True
                    else:
                        if (
                            self._press_action(action_name)
                            and self._release_action(action_name)
                        ):
                            self._prev[action_name] = True
            else:
                if was_active:
                    if self._grace.get(action_name, 0) == 0:
                        self._grace[action_name] = self._grace_frames
                    else:
                        self._grace[action_name] -= 1
                    if self._grace.get(action_name, 0) == 0:
                        if mode is ActionMode.CONTINUOUS:
                            if self._release_action(action_name):
                                self._prev[action_name] = False
                        else:
                            self._prev[action_name] = False



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
        dispatcher = TetrisKeyboardDispatcher(keyboard)

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
                        dispatcher.draw_pressed_keys_overlay(img, dispatcher, keyboard)

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
