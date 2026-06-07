import sys
import re

filepath = r'c:\Users\roney\Downloads\tetris\tetris\HandGestures.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Remove try/except block
content = re.sub(
    r'try:\n    from tetris.HandTrackingModule import HandData, HandTracker\nexcept ModuleNotFoundError:\n    from HandTrackingModule import HandData, HandTracker',
    r'from tetris.HandTrackingModule import HandData, HandTracker',
    content
)

# 2. Add HandState dataclass definition
dataclass_def = '''@dataclass
class HandState:
    smoothed_wrist: tuple[float, float] | None = None
    y_history: deque[float] = field(default_factory=lambda: deque(maxlen=2))
    prev_y: float | None = None
    prev_wrist_x: float | None = None
    hard_drop_cooldown: int = 0
    swipe_drop_cooldown: int = 0
    prev_wrist: tuple[float, float] | None = None
    wrist_velocities: deque[float] = field(default_factory=lambda: deque(maxlen=2))
    curled_count: int = 0
    finger_curled: tuple[bool, ...] = ()
    last_gesture: Gesture = Gesture.NONE
    gesture_settle_cooldown: int = 0
    swipe_confirm_count: int = 0
    fist_confirm_count: int = 0
    peace_confirm_count: int = 0
    rock_confirm_count: int = 0
    active_gesture_frames: int = 0

'''

# Insert after `GestureState` definition
content = content.replace('def actions_for_gesture(gesture: Gesture) -> dict[str, bool]:', dataclass_def + 'def actions_for_gesture(gesture: Gesture) -> dict[str, bool]:')

# 3. Modify init
init_repl = '''        self._gesture_motion_gate = gesture_motion_gate

        self._hand_states: dict[HandKey, HandState] = {}
        
        self._hard_drop_cooldown_frames: int = hard_drop_cooldown
        self._swipe_drop_cooldown_frames: int = 12
        self._gesture_settle_cooldown_frames: int = 5
        self._swipe_confirm_needed: int = 1
        self._fist_confirm_needed: int = 4
        self._peace_confirm_needed: int = 5
        self._rock_confirm_needed: int = 5

        # Adaptive deadzone: opposite-axis gate tightens the longer an action is held.'''

content = re.sub(r'        self\._gesture_motion_gate = gesture_motion_gate.*?# Adaptive deadzone:', init_repl + '\n        # Adaptive deadzone:', content, flags=re.DOTALL)

# 4. Modify cleanup
cleanup_repl = '''    def _cleanup_missing_hands(self, present: set[HandKey]) -> None:
        for hand_key in list(self._hand_states.keys()):
            if hand_key not in present:
                self._hand_states.pop(hand_key, None)

    def reset(self) -> None:
        self._hand_states.clear()'''

content = re.sub(r'    def _cleanup_missing_hands\(self, present: set\[HandKey\]\) -> None:.*?        self\._active_gesture_frames\.clear\(\)', cleanup_repl, content, flags=re.DOTALL)

# 5. Helper to get state
content = content.replace('    def _deadzone_scale(self, hand_key: HandKey) -> float:', '    def _get_state(self, hand_key: HandKey) -> HandState:\n        if hand_key not in self._hand_states:\n            self._hand_states[hand_key] = HandState()\n        return self._hand_states[hand_key]\n\n    def _deadzone_scale(self, hand_key: HandKey) -> float:')

# 6. Replace usages in the rest of the file
reps = [
    ('self._active_gesture_frames.get(hand_key, 0)', 'self._get_state(hand_key).active_gesture_frames'),
    ('self._active_gesture_frames[hand_key]', 'self._get_state(hand_key).active_gesture_frames'),
    ('self._last_hand_gesture.get(hand_key, Gesture.NONE)', 'self._get_state(hand_key).last_gesture'),
    ('self._gesture_settle_cooldown.get(hand_key, 0)', 'self._get_state(hand_key).gesture_settle_cooldown'),
    ('self._gesture_settle_cooldown[hand_key]', 'self._get_state(hand_key).gesture_settle_cooldown'),
    ('self._last_hand_gesture[hand_key]', 'self._get_state(hand_key).last_gesture'),
    ('self._curled_count[hand_key]', 'self._get_state(hand_key).curled_count'),
    ('self._curled_count.get(hand_key, 0)', 'self._get_state(hand_key).curled_count'),
    ('self._finger_curled[hand_key]', 'self._get_state(hand_key).finger_curled'),
    ('self._finger_curled.get(hand_key)', 'self._get_state(hand_key).finger_curled'),
    ('self._fist_confirm_count.get(hand_key, 0)', 'self._get_state(hand_key).fist_confirm_count'),
    ('self._fist_confirm_count[hand_key]', 'self._get_state(hand_key).fist_confirm_count'),
    ('self._peace_confirm_count.get(hand_key, 0)', 'self._get_state(hand_key).peace_confirm_count'),
    ('self._peace_confirm_count[hand_key]', 'self._get_state(hand_key).peace_confirm_count'),
    ('self._rock_confirm_count.get(hand_key, 0)', 'self._get_state(hand_key).rock_confirm_count'),
    ('self._rock_confirm_count[hand_key]', 'self._get_state(hand_key).rock_confirm_count'),
    ('self._swipe_confirm_count.get(hand_key, 0)', 'self._get_state(hand_key).swipe_confirm_count'),
    ('self._swipe_confirm_count[hand_key]', 'self._get_state(hand_key).swipe_confirm_count'),
    ('self._smoothed_wrist.get(hand_key)', 'self._get_state(hand_key).smoothed_wrist'),
    ('self._smoothed_wrist[hand_key]', 'self._get_state(hand_key).smoothed_wrist'),
    ('self._prev_wrist_x.get(hand_key)', 'self._get_state(hand_key).prev_wrist_x'),
    ('self._prev_wrist_x[hand_key]', 'self._get_state(hand_key).prev_wrist_x'),
    ('self._prev_wrist.get(hand_key)', 'self._get_state(hand_key).prev_wrist'),
    ('self._prev_wrist[hand_key]', 'self._get_state(hand_key).prev_wrist'),
    ('hand_key in self._smoothed_wrist', 'self._get_state(hand_key).smoothed_wrist is not None'),
    ('hand_key in self._prev_wrist and self._prev_wrist[hand_key] is not None', 'self._get_state(hand_key).prev_wrist is not None'),
    ('self._y_history.get(hand_key)', 'self._get_state(hand_key).y_history'),
    ('self._y_history[hand_key]', 'self._get_state(hand_key).y_history'),
    ('self._prev_y.get(hand_key)', 'self._get_state(hand_key).prev_y'),
    ('self._prev_y[hand_key]', 'self._get_state(hand_key).prev_y'),
    ('self._hard_drop_cooldown.get(hand_key, 0)', 'self._get_state(hand_key).hard_drop_cooldown'),
    ('self._hard_drop_cooldown[hand_key]', 'self._get_state(hand_key).hard_drop_cooldown'),
    ('self._swipe_drop_cooldown.get(hand_key, 0)', 'self._get_state(hand_key).swipe_drop_cooldown'),
    ('self._swipe_drop_cooldown[hand_key]', 'self._get_state(hand_key).swipe_drop_cooldown'),
    ('self._wrist_velocities.get(hand_key)', 'self._get_state(hand_key).wrist_velocities'),
    ('self._wrist_velocities[hand_key]', 'self._get_state(hand_key).wrist_velocities'),
    ('hand_key not in self._wrist_velocities', 'len(self._get_state(hand_key).wrist_velocities) == 0'), # Approximate, actually deque is created empty
    ('hand_key not in self._y_history', 'len(self._get_state(hand_key).y_history) == 0 and self._get_state(hand_key).prev_y is None'), # Approximate
]

for old, new in reps:
    content = content.replace(old, new)

content = re.sub(r'def main\(.*', '', content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(content)
