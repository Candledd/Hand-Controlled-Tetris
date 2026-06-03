from __future__ import annotations

from enum import Enum

from GestureConversions import (
    ACTION_KEY_MAP,
    GestureKeyboardDispatcher,
    KeyboardKey,
    Win32Keyboard,
)
from HandGestures import (
    ACTION_HARD_DROP,
    ACTION_LEFT,
    ACTION_PAUSE,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_SOFT_DROP,
    GestureState,
)


class ActionMode(Enum):
    """How a gesture action is delivered to the OS keyboard layer.

    CONTINUOUS
        Key is held down for the entire duration of the gesture: press
        on rising edge, release on falling edge. The OS's built-in
        autorepeat produces repeated key events, which is what movement-
        style inputs (LEFT, RIGHT, SOFT_DROP) want.

    SINGLE
        One press+release is fired on the rising edge. The OS sees a
        discrete key tap regardless of how long the gesture is held.
        Falling edge is a no-op (the key was already released on the
        rising edge), so re-arming only happens when the gesture ends
        and starts again.
    """

    CONTINUOUS = "continuous"
    SINGLE = "single"


TETRIS_ACTION_MODES: dict[str, ActionMode] = {
    ACTION_LEFT: ActionMode.CONTINUOUS,
    ACTION_RIGHT: ActionMode.CONTINUOUS,
    ACTION_SOFT_DROP: ActionMode.CONTINUOUS,
    ACTION_ROTATE: ActionMode.SINGLE,
    ACTION_HARD_DROP: ActionMode.SINGLE,
    ACTION_PAUSE: ActionMode.CONTINUOUS,
}


class TetrisKeyboardDispatcher(GestureKeyboardDispatcher):
    """GestureKeyboardDispatcher variant with per-action CONTINUOUS/SINGLE policy.

    Inherits the base dispatcher's keyboard backend, action-to-key map,
    edge-triggered bookkeeping (`_known`, `_prev`), and the exit-path
    `release_all`. Overrides only `dispatch` so each action's delivery
    mode (held vs. one-shot) is honoured independently.

    The mode table is supplied at construction time and is not mutated
    after that. Passing a partial dict falls back to `CONTINUOUS` for
    any unlisted action, matching the base dispatcher's hold-the-key
    semantics for the gap.
    """

    def __init__(
        self,
        keyboard: Win32Keyboard,
        action_key_map: dict[str, tuple[KeyboardKey, ...]] = ACTION_KEY_MAP,
        action_modes: dict[str, ActionMode] = TETRIS_ACTION_MODES,
    ) -> None:
        super().__init__(keyboard, action_key_map=action_key_map)
        self._modes: dict[str, ActionMode] = dict(action_modes)

    def dispatch(self, state: GestureState) -> None:
        """Update held or sent keys to match the current action set.

        CONTINUOUS actions use the base dispatcher's press-on-rising,
        release-on-falling logic, so the OS sees a held key and its
        built-in autorepeat produces repeated events.

        SINGLE actions fire a press+release on the rising edge and
        then become inert for the rest of the gesture. The falling
        edge clears the internal latch so the next rising edge fires
        a fresh tap. This avoids the autorepeat that a held key would
        produce, which is wrong for discrete actions like rotate and
        hard-drop.

        Partial-failure handling matches the base dispatcher: an OS
        call that returns False leaves `self._prev[action_name]` at
        its prior value, so the next frame retries the transition
        instead of silently diverging from the OS.
        """
        current = state.actions
        for action_name in self._known:
            was_active = self._prev.get(action_name, False)
            is_active = current.get(action_name, False)
            mode = self._modes.get(action_name, ActionMode.CONTINUOUS)
            if mode is ActionMode.CONTINUOUS:
                if is_active and not was_active:
                    if self._press_action(action_name):
                        self._prev[action_name] = True
                elif was_active and not is_active:
                    if self._release_action(action_name):
                        self._prev[action_name] = False
            else:
                if is_active and not was_active:
                    if (
                        self._press_action(action_name)
                        and self._release_action(action_name)
                    ):
                        self._prev[action_name] = True
                elif was_active and not is_active:
                    self._prev[action_name] = False
