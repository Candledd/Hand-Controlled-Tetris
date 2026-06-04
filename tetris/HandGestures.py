from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import cv2
import time
import numpy as np

try:
    from tetris.HandTrackingModule import HandData, HandTracker
except ModuleNotFoundError:
    from HandTrackingModule import HandData, HandTracker

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
MIDDLE_TIP = 12
PINKY_MCP = 17
HandKey: TypeAlias = str

# Used only for closed-hand (fist) detection — not per-finger gesture tracking.
_FIST_TIP_MCP_PAIRS = (
    (4, 2),   # thumb tip → thumb base
    (8, 5),   # index
    (12, 9),  # middle
    (16, 13), # ring
    (20, 17), # pinky
)

# Peace sign: thumb, ring, and pinky tips must be clustered together.
_PEACE_TIP_INDICES = (4, 16, 20)

# Rock sign: thumb, middle, and ring tips must be clustered together.
_ROCK_TIP_INDICES = (4, 12, 16)

ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_ROTATE = "rotate"
ACTION_HARD_DROP = "hard_drop"
ACTION_SOFT_DROP = "soft_drop"
ACTION_RETRY = "retry"
ACTION_HOLD = "hold"

ALL_ACTIONS: tuple[str, ...] = (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_HARD_DROP,
    ACTION_SOFT_DROP,
    ACTION_RETRY,
    ACTION_HOLD,
)


class Gesture(Enum):
    NONE = "none"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    FIST = "fist"
    SWIPE_DOWN_FAST = "swipe_down_fast"
    SWIPE_DOWN_SLOW = "swipe_down_slow"
    PEACE = "peace"
    ROCK = "rock"


GESTURE_ACTIONS: dict[Gesture, str] = {
    Gesture.SWIPE_LEFT: ACTION_LEFT,
    Gesture.SWIPE_RIGHT: ACTION_RIGHT,
    Gesture.FIST: ACTION_ROTATE,
    Gesture.SWIPE_DOWN_FAST: ACTION_HARD_DROP,
    Gesture.SWIPE_DOWN_SLOW: ACTION_SOFT_DROP,
    Gesture.PEACE: ACTION_HOLD,
    Gesture.ROCK: ACTION_RETRY,
}


@dataclass
class GestureState:
    gesture: Gesture = Gesture.NONE
    actions: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in ALL_ACTIONS}
    )

    @property
    def active_actions(self) -> list[str]:
        return [name for name, pressed in self.actions.items() if pressed]


def actions_for_gesture(gesture: Gesture) -> dict[str, bool]:
    """Action dict for a single gesture with all others off."""
    actions = {name: False for name in ALL_ACTIONS}
    action = GESTURE_ACTIONS.get(gesture)
    if action:
        actions[action] = True
    return actions


def draw_gesture_overlay(img: np.ndarray, state: GestureState) -> None:
    """Draw overlay: Gesture + Action."""
    h, w = img.shape[:2]
    cv2.putText(
        img,
        f"Gesture: {state.gesture.value}",
        (int(w * 0.02), int(h * 0.08)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    label = ", ".join(state.active_actions) if state.active_actions else "-"
    cv2.putText(
        img,
        f"Actions: {label}",
        (int(w * 0.02), int(h * 0.12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


class GestureDetector:
    """Turns a list of `HandData` into a per-frame `GestureState`."""

    def __init__(
        self,
        fist_curled_threshold: float = 0.09,
        min_curled_fingers: int = 5,
        hard_drop_velocity: float = 0.03,
        soft_drop_threshold: float = 0.06,
        drop_velocity_divisor: int = 6,
        hard_drop_cooldown: int = 15,
        swipe_threshold: float = 0.075,
        swipe_velocity_divisor: int = 5,
        z_weight: float = 2.0,
        palm_tilt_threshold: float = 0.35,
        ema_alpha: float = 0.5,
        curled_tip_proximity: float = 0.15,
    ) -> None:
        self.fist_curled_threshold = fist_curled_threshold
        self._fist_curled_threshold_sq = fist_curled_threshold ** 2
        self.min_curled_fingers = min_curled_fingers
        self.hard_drop_velocity = hard_drop_velocity
        self.soft_drop_threshold = soft_drop_threshold
        self.swipe_threshold = swipe_threshold
        self._z_weight = z_weight
        self._palm_tilt_threshold = palm_tilt_threshold

        # alpha=1.0 means no smoothing (raw), alpha=0.0 means fully lagged.
        self._ema_alpha = ema_alpha
        self._curled_tip_proximity_sq = curled_tip_proximity ** 2
        self._smoothed_wrist: dict[HandKey, tuple[float, float]] = {}

        self._y_history: dict[HandKey, deque[float]] = {}
        self._prev_y: dict[HandKey, float | None] = {}
        self._prev_wrist_x: dict[HandKey, float | None] = {}
        self._hard_drop_cooldown: dict[HandKey, int] = {}
        self._hard_drop_cooldown_frames: int = hard_drop_cooldown
        self._swipe_drop_cooldown: dict[HandKey, int] = {}
        self._swipe_drop_cooldown_frames: int = 12
        self._prev_wrist: dict[HandKey, tuple[float, float] | None] = {}
        self._wrist_velocities: dict[HandKey, deque[float]] = {}
        self._curled_count: dict[HandKey, int] = {}
        self._finger_curled: dict[HandKey, tuple[bool, ...]] = {}
        self._last_hand_gesture: dict[HandKey, Gesture] = {}
        self._gesture_settle_cooldown: dict[HandKey, int] = {}
        self._gesture_settle_cooldown_frames: int = 5
        self._swipe_confirm_count: dict[HandKey, int] = {}
        self._swipe_confirm_needed: int = 1
        self._fist_confirm_count: dict[HandKey, int] = {}
        self._fist_confirm_needed: int = 3
        self._peace_confirm_count: dict[HandKey, int] = {}
        self._peace_confirm_needed: int = 3
        self._rock_confirm_count: dict[HandKey, int] = {}
        self._rock_confirm_needed: int = 5

        # Adaptive deadzone: opposite-axis gate tightens the longer an action is held.
        self._active_gesture_frames: dict[HandKey, int] = {}
        self._deadzone_max_frames: int = 20
        self._deadzone_max_scale: float = 3.0

        # Opposite-axis gates: strong motion on the wrong axis suppresses
        # the current detector (e.g. a wrist dip mid-swipe shouldn't drop).
        self._horizontal_gate = swipe_threshold / swipe_velocity_divisor * 1.5
        self._vertical_gate = soft_drop_threshold / drop_velocity_divisor * 2.0

        self._swipe_velocity_threshold = swipe_threshold / swipe_velocity_divisor * 0.39
        self._soft_drop_velocity_threshold = soft_drop_threshold / drop_velocity_divisor * 0.5

    def __enter__(self) -> "GestureDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    @staticmethod
    def draw_gesture_overlay(img: np.ndarray, state: GestureState) -> None:
        draw_gesture_overlay(img, state)

    def update(self, hands: list[HandData]) -> GestureState:
        """Run per-frame detection and return the resulting state."""
        present = self._present_hand_keys(hands)
        self._cleanup_missing_hands(present)

        if not hands:
            return GestureState(actions=actions_for_gesture(Gesture.NONE))

        combined_actions = {name: False for name in ALL_ACTIONS}
        last_gesture = Gesture.NONE
        seen_hands: Counter[str] = Counter()

        for hand in hands:
            hand_key = self._hand_key(hand, seen_hands)
            self._update_finger_states(hand, hand_key)
            raw_x = hand.landmarks_norm[WRIST][0]
            raw_y = hand.landmarks_norm[WRIST][1]

            # EMA-smooth the wrist. Keyed by handedness (not list index)
            # so history follows the hand when detection order flips.
            if hand_key in self._smoothed_wrist:
                prev_sx, prev_sy = self._smoothed_wrist[hand_key]
                a = self._ema_alpha
                wrist_x = a * raw_x + (1.0 - a) * prev_sx
                wrist_y = a * raw_y + (1.0 - a) * prev_sy
            else:
                wrist_x = raw_x
                wrist_y = raw_y
            self._smoothed_wrist[hand_key] = (wrist_x, wrist_y)

            prev_wrist_x = self._prev_wrist_x.get(hand_key)
            self._prev_wrist_x[hand_key] = wrist_x

            gesture = self._detect_drop(hand, hand_key, wrist_x, wrist_y, prev_wrist_x)
            if gesture is Gesture.NONE:
                gesture = self._detect_fist(hand, hand_key)
            if gesture is Gesture.NONE:
                gesture = self._detect_peace(hand, hand_key)
            if gesture is Gesture.NONE:
                gesture = self._detect_rock(hand, hand_key)
            if gesture is Gesture.NONE:
                gesture = self._detect_swipe(hand, hand_key, wrist_x, prev_wrist_x)

            gesture = self._debounce_gesture(hand_key, gesture)

            if gesture is not Gesture.NONE:
                self._active_gesture_frames[hand_key] = (
                    self._active_gesture_frames.get(hand_key, 0) + 1
                )
            else:
                self._active_gesture_frames[hand_key] = 0

            if gesture is not Gesture.NONE:
                if last_gesture is Gesture.NONE:
                    last_gesture = gesture
                action = GESTURE_ACTIONS.get(gesture)
                if action:
                    combined_actions[action] = True

        return GestureState(
            gesture=last_gesture,
            actions=combined_actions,
        )

    def _present_hand_keys(self, hands: list[HandData]) -> set[HandKey]:
        seen_hands: Counter[str] = Counter()
        return {self._hand_key(hand, seen_hands) for hand in hands}

    @staticmethod
    def _hand_key(hand: HandData, seen_hands: Counter[str]) -> HandKey:
        label = hand.handedness or "Unknown"
        seen_hands[label] += 1
        return f"{label}:{seen_hands[label]}"

    def _cleanup_missing_hands(self, present: set[HandKey]) -> None:
        for hand_key in list(self._y_history.keys()):
            if hand_key in present:
                continue
            self._y_history.pop(hand_key, None)
            self._prev_y.pop(hand_key, None)
            self._prev_wrist_x.pop(hand_key, None)
            self._hard_drop_cooldown.pop(hand_key, None)
            self._swipe_drop_cooldown.pop(hand_key, None)
            self._prev_wrist.pop(hand_key, None)
            self._wrist_velocities.pop(hand_key, None)
            self._curled_count.pop(hand_key, None)
            self._finger_curled.pop(hand_key, None)
            self._last_hand_gesture.pop(hand_key, None)
            self._gesture_settle_cooldown.pop(hand_key, None)
            self._swipe_confirm_count.pop(hand_key, None)
            self._fist_confirm_count.pop(hand_key, None)
            self._peace_confirm_count.pop(hand_key, None)
            self._rock_confirm_count.pop(hand_key, None)
            self._smoothed_wrist.pop(hand_key, None)
            self._active_gesture_frames.pop(hand_key, None)

    def reset(self) -> None:
        self._y_history.clear()
        self._prev_y.clear()
        self._prev_wrist_x.clear()
        self._hard_drop_cooldown.clear()
        self._swipe_drop_cooldown.clear()
        self._prev_wrist.clear()
        self._wrist_velocities.clear()
        self._curled_count.clear()
        self._finger_curled.clear()
        self._last_hand_gesture.clear()
        self._gesture_settle_cooldown.clear()
        self._swipe_confirm_count.clear()
        self._fist_confirm_count.clear()
        self._peace_confirm_count.clear()
        self._rock_confirm_count.clear()
        self._smoothed_wrist.clear()
        self._active_gesture_frames.clear()

    def _deadzone_scale(self, hand_key: HandKey) -> float:
        """Return a multiplier >= 1.0 that grows with sustained gesture duration."""
        frames = self._active_gesture_frames.get(hand_key, 0)
        t = min(frames / self._deadzone_max_frames, 1.0)
        return 1.0 + t * (self._deadzone_max_scale - 1.0)

    def _debounce_gesture(self, hand_key: HandKey, gesture: Gesture) -> Gesture:
        prev = self._last_hand_gesture.get(hand_key, Gesture.NONE)
        cooldown = self._gesture_settle_cooldown.get(hand_key, 0)

        if cooldown > 0:
            self._gesture_settle_cooldown[hand_key] = cooldown - 1

        if gesture != Gesture.NONE and prev == Gesture.NONE and cooldown > 0:
            if gesture != Gesture.SWIPE_DOWN_FAST:
                gesture = Gesture.NONE

        if gesture != Gesture.NONE:
            self._gesture_settle_cooldown[hand_key] = self._gesture_settle_cooldown_frames

        self._last_hand_gesture[hand_key] = gesture
        return gesture

    def _update_finger_states(self, hand: HandData, hand_key: HandKey) -> None:
        curled_count = 0
        states = []
        for tip_idx, mcp_idx in _FIST_TIP_MCP_PAIRS:
            tip = hand.landmarks_norm[tip_idx]
            mcp = hand.landmarks_norm[mcp_idx]
            dx = tip[0] - mcp[0]
            dy = tip[1] - mcp[1]
            dz = (tip[2] - mcp[2]) * self._z_weight
            is_curled = dx * dx + dy * dy + dz * dz < self._fist_curled_threshold_sq
            states.append(is_curled)
            if is_curled:
                curled_count += 1
        self._curled_count[hand_key] = curled_count
        self._finger_curled[hand_key] = tuple(states)

    @staticmethod
    def _compute_palm_normal_z(hand: HandData) -> float:
        """z-component of the palm-plane normal (|result| near 1 = palm faces camera)."""
        w = hand.landmarks_norm[WRIST]
        idx = hand.landmarks_norm[INDEX_MCP]
        pnk = hand.landmarks_norm[PINKY_MCP]

        ax, ay, az = idx[0] - w[0], idx[1] - w[1], idx[2] - w[2]
        bx, by, bz = pnk[0] - w[0], pnk[1] - w[1], pnk[2] - w[2]

        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag < 1e-9:
            return 1.0  # degenerate: assume facing camera

        return nz / mag

    def _detect_fist(self, hand: HandData, hand_key: HandKey) -> Gesture:
        wrist = hand.landmarks_norm[WRIST]
        wrist_x, wrist_y = wrist[0], wrist[1]

        # Upright cone gate (±45°). Suppress if the hand is horizontal
        # or pointing down (rotation otherwise reads as a fist).
        mid_mcp = hand.landmarks_norm[MIDDLE_MCP]
        up_dy = wrist_y - mid_mcp[1]
        up_dx = mid_mcp[0] - wrist_x
        if up_dy < abs(up_dx):
            self._fist_confirm_count[hand_key] = 0
            return Gesture.NONE

        min_needed = self.min_curled_fingers

        # Edge-on palm: 2D foreshortening curls extended fingers,
        # so demand all 5 curled before counting it.
        palm_nz = abs(self._compute_palm_normal_z(hand))
        if palm_nz < self._palm_tilt_threshold:
            min_needed = 5

        if hand_key not in self._wrist_velocities:
            self._wrist_velocities[hand_key] = deque(maxlen=2)

        if hand_key in self._prev_wrist and self._prev_wrist[hand_key] is not None:
            prev_x, prev_y = self._prev_wrist[hand_key]
            velocity = abs(wrist_x - prev_x) + abs(wrist_y - prev_y)
            self._wrist_velocities[hand_key].append(velocity)
            avg_v = sum(self._wrist_velocities[hand_key]) / len(
                self._wrist_velocities[hand_key]
            )
            if avg_v >= self._soft_drop_velocity_threshold * 2:
                min_needed = max(min_needed, self.min_curled_fingers + 1)

        self._prev_wrist[hand_key] = (wrist_x, wrist_y)

        if self._curled_count.get(hand_key, 0) >= min_needed:
            count = self._fist_confirm_count.get(hand_key, 0) + 1
            self._fist_confirm_count[hand_key] = count
            if count >= self._fist_confirm_needed:
                return Gesture.FIST
            return Gesture.NONE

        self._fist_confirm_count[hand_key] = 0
        return Gesture.NONE

    def _detect_peace(self, hand: HandData, hand_key: HandKey) -> Gesture:
        states = self._finger_curled.get(hand_key)
        if not states or len(states) != 5:
            return Gesture.NONE

        # Index + middle up, thumb/ring/pinky curled.
        if not (states[0] and not states[1] and not states[2] and states[3] and states[4]):
            return Gesture.NONE

        # The three curled tips must be clustered together (peace: thumb tucked over the ring/pinky).
        tips = [hand.landmarks_norm[i] for i in _PEACE_TIP_INDICES]
        zw = self._z_weight
        thresh = self._curled_tip_proximity_sq
        for a, b in ((0, 1), (0, 2), (1, 2)):
            dx = tips[a][0] - tips[b][0]
            dy = tips[a][1] - tips[b][1]
            dz = (tips[a][2] - tips[b][2]) * zw
            if dx * dx + dy * dy + dz * dz >= thresh:
                return Gesture.NONE
        return Gesture.PEACE

    def _detect_rock(self, hand: HandData, hand_key: HandKey) -> Gesture:
        states = self._finger_curled.get(hand_key)
        if not states or len(states) != 5:
            return Gesture.NONE

        # Index + pinky up, thumb/middle/ring curled.
        if not (states[0] and not states[1] and states[2] and states[3] and not states[4]):
            return Gesture.NONE

        # The three curled tips must be clustered together. (thumb tucked over middle/ring)
        tips = [hand.landmarks_norm[i] for i in _ROCK_TIP_INDICES]
        zw = self._z_weight
        thresh = self._curled_tip_proximity_sq
        for a, b in ((0, 1), (0, 2), (1, 2)):
            dx = tips[a][0] - tips[b][0]
            dy = tips[a][1] - tips[b][1]
            dz = (tips[a][2] - tips[b][2]) * zw
            if dx * dx + dy * dy + dz * dz >= thresh:
                return Gesture.NONE
        return Gesture.ROCK

    def _detect_swipe(self, hand: HandData, hand_key: HandKey,
                      hand_wrist_x: float, prev_wrist_x: float | None) -> Gesture:
        # Suppress if the hand is dropping — y-history comes from _detect_drop.
        y_history = self._y_history.get(hand_key)
        if y_history is not None and len(y_history) >= 2:
            y_velocity = abs(y_history[-1] - y_history[-2])
            if y_velocity >= self._vertical_gate / self._deadzone_scale(hand_key):
                self._swipe_confirm_count[hand_key] = 0
                return Gesture.NONE

        if self._curled_count.get(hand_key, 0) >= 2:
            self._swipe_confirm_count[hand_key] = 0
            return Gesture.NONE

        if prev_wrist_x is not None:
            velocity = hand_wrist_x - prev_wrist_x
            is_right = hand.handedness == "Right"

            scaled_swipe_threshold = self._swipe_velocity_threshold * self._deadzone_scale(hand_key)

            candidate = Gesture.NONE
            if is_right and velocity >= scaled_swipe_threshold:
                candidate = Gesture.SWIPE_LEFT
            elif not is_right and velocity <= -scaled_swipe_threshold:
                candidate = Gesture.SWIPE_RIGHT

            if candidate is not Gesture.NONE:
                count = self._swipe_confirm_count.get(hand_key, 0) + 1
                self._swipe_confirm_count[hand_key] = count
                if count >= self._swipe_confirm_needed:
                    self._swipe_confirm_count[hand_key] = 0
                    self._swipe_drop_cooldown[hand_key] = self._swipe_drop_cooldown_frames
                    return candidate
                return Gesture.NONE

            self._swipe_confirm_count[hand_key] = 0
            return Gesture.NONE

        return Gesture.NONE

    def _detect_drop(self, hand: HandData, hand_key: HandKey,
                     hand_wrist_x: float, hand_wrist_y: float,
                     prev_wrist_x: float | None) -> Gesture:
        if hand_key not in self._y_history:
            self._y_history[hand_key] = deque(maxlen=2)
            self._prev_y[hand_key] = None
            self._hard_drop_cooldown[hand_key] = 0
            self._swipe_drop_cooldown[hand_key] = 0

        average_y = hand_wrist_y
        average_x = hand_wrist_x
        history = self._y_history[hand_key]

        if self._hard_drop_cooldown[hand_key] > 0:
            self._hard_drop_cooldown[hand_key] -= 1
            history.append(average_y)
            self._prev_y[hand_key] = average_y
            return Gesture.NONE

        # Hard drop is checked before the horizontal gate so a deliberate
        # downward flick isn't suppressed by mild sideways drift.
        if self._prev_y[hand_key] is not None:
            velocity = average_y - self._prev_y[hand_key]

            if velocity >= self.hard_drop_velocity:
                history.append(average_y)
                self._prev_y[hand_key] = average_y
                self._hard_drop_cooldown[hand_key] = self._hard_drop_cooldown_frames
                self._swipe_drop_cooldown[hand_key] = 0
                return Gesture.SWIPE_DOWN_FAST

        if prev_wrist_x is not None:
            x_velocity = abs(average_x - prev_wrist_x)
            if x_velocity >= self._horizontal_gate / self._deadzone_scale(hand_key):
                history.append(average_y)
                self._prev_y[hand_key] = average_y
                return Gesture.NONE

        if self._prev_y[hand_key] is not None:
            if self._swipe_drop_cooldown[hand_key] > 0:
                self._swipe_drop_cooldown[hand_key] -= 1
                self._prev_y[hand_key] = average_y
                history.append(average_y)
                return Gesture.NONE

            if self._curled_count.get(hand_key, 0) >= 2:
                self._prev_y[hand_key] = average_y
                history.append(average_y)
                return Gesture.NONE

            velocity = average_y - self._prev_y[hand_key]
            scaled_drop_threshold = self._soft_drop_velocity_threshold * self._deadzone_scale(hand_key)
            if velocity >= scaled_drop_threshold:
                history.append(average_y)
                self._prev_y[hand_key] = average_y
                return Gesture.SWIPE_DOWN_SLOW

        self._prev_y[hand_key] = average_y
        history.append(average_y)
        return Gesture.NONE


def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    prev_time = 0.0
    detector = GestureDetector()

    try:
        with HandTracker(max_hands=2) as tracker:
            while True:
                success, img = cap.read()
                if not success:
                    break

                img, hands = tracker.find_hands(img)
                img = tracker.label_hands(img)
                state = detector.update(hands)
                draw_gesture_overlay(img, state)

                curr_time = time.time()
                dt = curr_time - prev_time
                fps = int(1 / dt) if dt > 1e-6 else 0
                prev_time = curr_time
                cv2.putText(
                    img,
                    str(fps),
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Hand Gestures", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
