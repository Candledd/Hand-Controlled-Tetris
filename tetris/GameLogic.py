from __future__ import annotations

from enum import Enum
import cv2

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
    ACTION_ROTATE_CCW,
    ACTION_SOFT_DROP,
    GestureState,
    GestureDetector,
)
from tetris.HandTrackingModule import HandTracker

class ActionMode(Enum):
    CONTINUOUS = "continuous"
    SINGLE = "single"


TETRIS_ACTION_MODES: dict[str, ActionMode] = {
    ACTION_LEFT: ActionMode.CONTINUOUS,
    ACTION_RIGHT: ActionMode.CONTINUOUS,
    ACTION_SOFT_DROP: ActionMode.CONTINUOUS,
    ACTION_ROTATE: ActionMode.SINGLE,
    ACTION_ROTATE_CCW: ActionMode.SINGLE,
    ACTION_HARD_DROP: ActionMode.SINGLE,
    ACTION_RETRY: ActionMode.CONTINUOUS,
    ACTION_HOLD: ActionMode.SINGLE,
}


class TetrisKeyboardDispatcher(GestureKeyboardDispatcher):
    """Per-action mode + falling-edge grace on top of the base dispatcher."""
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
        """Update held/sent keys to match the current action set.

        CONTINUOUS actions use press-on-rising / release-on-falling so
        OS autorepeat produces repeated events.
        SINGLE actions tap press+release on the rising edge and stay
        inert until the falling edge clears the latch, avoiding the
        autorepeat that a held key would produce.
        Both modes defer release / latch-clear by up to `grace_frames`
        so a single false-negative frame mid-slide doesn't break
        autorepeat or cause a spurious re-fire.
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


