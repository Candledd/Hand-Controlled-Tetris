from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import cv2
import time
import numpy as np

from HandTrackingModule import HandData, HandTracker

WRIST = 0
INDEX_MCP = 5
MIDDLE_TIP = 12
PINKY_MCP = 17

# Used only for closed-hand (fist) detection — not per-finger gesture tracking.
_FIST_TIP_MCP_PAIRS = (
    (4, 2),   # thumb tip → thumb base
    (8, 5),   # index
    (12, 9),  # middle
    (16, 13), # ring
    (20, 17), # pinky
)

ACTION_LEFT = "left"
ACTION_RIGHT = "right"
ACTION_ROTATE = "rotate"
ACTION_HARD_DROP = "hard_drop"
ACTION_SOFT_DROP = "soft_drop"
ACTION_PAUSE = "pause"

ALL_ACTIONS: tuple[str, ...] = (
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_HARD_DROP,
    ACTION_SOFT_DROP,
    ACTION_PAUSE,
)


class Gesture(Enum):
    NONE = "none"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    FIST = "fist"
    SWIPE_DOWN_FAST = "swipe_down_fast"
    SWIPE_DOWN_SLOW = "swipe_down_slow"


GESTURE_ACTIONS: dict[Gesture, str] = {
    Gesture.SWIPE_LEFT: ACTION_LEFT,
    Gesture.SWIPE_RIGHT: ACTION_RIGHT,
    Gesture.FIST: ACTION_ROTATE,
    Gesture.SWIPE_DOWN_FAST: ACTION_HARD_DROP,
    Gesture.SWIPE_DOWN_SLOW: ACTION_SOFT_DROP,
}


# Init
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
    """Ensure only one gesture is true, turns everything else false."""
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
    """Converts `HandData` lists into `GestureState` for frame."""

    def __init__(
        self,
        fist_curled_threshold: float = 0.09,
        min_curled_fingers: int = 4,
        hard_drop_velocity: float = 0.035,
        soft_drop_threshold: float = 0.06,
        drop_velocity_divisor: int = 6,
        hard_drop_cooldown: int = 15,
        swipe_threshold: float = 0.075,
        swipe_velocity_divisor: int = 5,
        z_weight: float = 2.0,
        palm_tilt_threshold: float = 0.35,
        ema_alpha: float = 0.5,
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
        self._smoothed_wrist: dict[int, tuple[float, float]] = {}

        self._y_history: dict[int, deque[float]] = {}
        self._prev_y: dict[int, float | None] = {}
        self._prev_wrist_x: dict[int, float | None] = {}
        self._hard_drop_cooldown: dict[int, int] = {}
        self._hard_drop_cooldown_frames: int = hard_drop_cooldown
        self._swipe_drop_cooldown: dict[int, int] = {}
        self._swipe_drop_cooldown_frames: int = 12
        self._prev_wrist: dict[int, tuple[float, float] | None] = {}
        self._wrist_velocities: dict[int, deque[float]] = {}
        self._curled_count: dict[int, int] = {}
        self._last_hand_gesture: dict[int, Gesture] = {}
        self._gesture_settle_cooldown: dict[int, int] = {}
        self._gesture_settle_cooldown_frames: int = 5
        self._swipe_confirm_count: dict[int, int] = {}
        self._swipe_confirm_needed: int = 1

        # Axis-dominance gates: if the opposite axis exceeds this per-frame
        # velocity, the detector suppresses itself.  Prevents wrist-extension
        # during swipes from triggering drops, and vice versa.
        self._horizontal_gate = swipe_threshold / swipe_velocity_divisor * 1.5
        self._vertical_gate = soft_drop_threshold / drop_velocity_divisor * 2.0

        # Per-frame velocity thresholds for continuous motion
        self._swipe_velocity_threshold = swipe_threshold / swipe_velocity_divisor * 0.39
        self._soft_drop_velocity_threshold = soft_drop_threshold / drop_velocity_divisor * 0.5

    def update(self, hands: list[HandData]) -> GestureState:
        """Looks for detection based on current frame."""
        # Clean up state for hands no longer present
        present = set(range(len(hands)))
        for hand_id in list(self._y_history.keys()):
            if hand_id not in present:
                self._y_history.pop(hand_id, None)
                self._prev_y.pop(hand_id, None)
                self._prev_wrist_x.pop(hand_id, None)
                self._hard_drop_cooldown.pop(hand_id, None)
                self._swipe_drop_cooldown.pop(hand_id, None)
                self._prev_wrist.pop(hand_id, None)
                self._wrist_velocities.pop(hand_id, None)
                self._curled_count.pop(hand_id, None)
                self._last_hand_gesture.pop(hand_id, None)
                self._gesture_settle_cooldown.pop(hand_id, None)
                self._swipe_confirm_count.pop(hand_id, None)
                self._smoothed_wrist.pop(hand_id, None)

        if not hands:
            return GestureState(actions=actions_for_gesture(Gesture.NONE))

        combined_actions = {name: False for name in ALL_ACTIONS}
        last_gesture = Gesture.NONE

        for hand_no, hand in enumerate(hands):
            self._update_finger_states(hand, hand_no)
            raw_x = hand.landmarks_norm[WRIST][0]
            raw_y = hand.landmarks_norm[WRIST][1]

            # Apply EMA smoothing to wrist position
            if hand_no in self._smoothed_wrist:
                prev_sx, prev_sy = self._smoothed_wrist[hand_no]
                a = self._ema_alpha
                wrist_x = a * raw_x + (1.0 - a) * prev_sx
                wrist_y = a * raw_y + (1.0 - a) * prev_sy
            else:
                wrist_x = raw_x
                wrist_y = raw_y
            self._smoothed_wrist[hand_no] = (wrist_x, wrist_y)

            prev_wrist_x = self._prev_wrist_x.get(hand_no)
            self._prev_wrist_x[hand_no] = wrist_x

            gesture = self._detect_drop(hand, hand_no, wrist_x, wrist_y, prev_wrist_x)
            if gesture is Gesture.NONE:
                gesture = self._detect_fist(hand, hand_no)
            if gesture is Gesture.NONE:
                gesture = self._detect_swipe(hand, hand_no, wrist_x, prev_wrist_x)

            gesture = self._debounce_gesture(hand_no, gesture)

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

    def reset(self) -> None:
        self._y_history.clear()
        self._prev_y.clear()
        self._prev_wrist_x.clear()
        self._hard_drop_cooldown.clear()
        self._swipe_drop_cooldown.clear()
        self._prev_wrist.clear()
        self._wrist_velocities.clear()
        self._curled_count.clear()
        self._last_hand_gesture.clear()
        self._gesture_settle_cooldown.clear()
        self._swipe_confirm_count.clear()
        self._smoothed_wrist.clear()

    def _debounce_gesture(self, hand_no: int, gesture: Gesture) -> Gesture:
        prev = self._last_hand_gesture.get(hand_no, Gesture.NONE)
        cooldown = self._gesture_settle_cooldown.get(hand_no, 0)

        if cooldown > 0:
            self._gesture_settle_cooldown[hand_no] = cooldown - 1

        if gesture != Gesture.NONE and prev == Gesture.NONE and cooldown > 0:
            if gesture != Gesture.SWIPE_DOWN_FAST:
                gesture = Gesture.NONE

        if gesture != Gesture.NONE:
            self._gesture_settle_cooldown[hand_no] = self._gesture_settle_cooldown_frames

        self._last_hand_gesture[hand_no] = gesture
        return gesture

    def _update_finger_states(self, hand: HandData, hand_no: int) -> None:
        curled_count = 0
        for tip_idx, mcp_idx in _FIST_TIP_MCP_PAIRS:
            tip = hand.landmarks_norm[tip_idx]
            mcp = hand.landmarks_norm[mcp_idx]
            dx = tip[0] - mcp[0]
            dy = tip[1] - mcp[1]
            dz = (tip[2] - mcp[2]) * self._z_weight
            if dx * dx + dy * dy + dz * dz < self._fist_curled_threshold_sq:
                curled_count += 1
        self._curled_count[hand_no] = curled_count

    @staticmethod
    def _compute_palm_normal_z(hand: HandData) -> float:
        """Compute the z-component of the palm plane normal vector.

        Uses the cross product of (WRIST → INDEX_MCP) × (WRIST → PINKY_MCP).
        Returns a value between -1.0 and 1.0:
          - |z| close to 1.0: palm faces the camera (hand is upright/vertical)
          - |z| close to 0.0: palm is edge-on (hand is tilted horizontally)
        """
        w = hand.landmarks_norm[WRIST]
        idx = hand.landmarks_norm[INDEX_MCP]
        pnk = hand.landmarks_norm[PINKY_MCP]

        # Vectors from wrist to index MCP and pinky MCP
        ax, ay, az = idx[0] - w[0], idx[1] - w[1], idx[2] - w[2]
        bx, by, bz = pnk[0] - w[0], pnk[1] - w[1], pnk[2] - w[2]

        # Cross product
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx

        # Magnitude
        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag < 1e-9:
            return 1.0  # Degenerate case — assume facing camera

        return nz / mag

    def _detect_fist(self, hand: HandData, hand_no: int) -> Gesture:
        wrist = hand.landmarks_norm[WRIST]
        wrist_x, wrist_y = wrist[0], wrist[1]

        min_needed = self.min_curled_fingers

        # ── Palm orientation gate ──
        # When the palm is significantly tilted (edge-on to the camera),
        # 2D foreshortening makes extended fingers look curled.
        # Require ALL 5 fingers to be curled when the palm is tilted.
        palm_nz = abs(self._compute_palm_normal_z(hand))
        if palm_nz < self._palm_tilt_threshold:
            min_needed = 5  # Must have every finger curled to count as fist

        # ── Smoothed motion gate ──
        # If the wrist is moving fast, raise the bar to avoid
        # transient curls during swipes.
        if hand_no not in self._wrist_velocities:
            self._wrist_velocities[hand_no] = deque(maxlen=2)

        if hand_no in self._prev_wrist and self._prev_wrist[hand_no] is not None:
            prev_x, prev_y = self._prev_wrist[hand_no]
            velocity = abs(wrist_x - prev_x) + abs(wrist_y - prev_y)
            self._wrist_velocities[hand_no].append(velocity)
            avg_v = sum(self._wrist_velocities[hand_no]) / len(self._wrist_velocities[hand_no])
            if avg_v >= self._soft_drop_velocity_threshold * 2:
                min_needed = max(min_needed, self.min_curled_fingers + 1)

        self._prev_wrist[hand_no] = (wrist_x, wrist_y)

        if self._curled_count.get(hand_no, 0) >= min_needed:
            return Gesture.FIST
        return Gesture.NONE

    def _detect_swipe(self, hand: HandData, hand_no: int,
                      hand_wrist_x: float, prev_wrist_x: float | None) -> Gesture:
        # NOTE: Swipe direction is intentionally mirrored.
        # MediaPipe reports "Right" for the hand on the right of the *camera's* frame,
        # but since the webcam image is flipped horizontally for the player,
        # a positive wrist_x velocity (moving right in camera-space) corresponds to
        # the player moving their hand LEFT on-screen.
        # Gate: if the hand is moving downward significantly (dropping),
        # suppress swipe detection (uses y-history from _detect_drop)
        y_history = self._y_history.get(hand_no)
        if y_history is not None and len(y_history) >= 2:
            y_velocity = abs(y_history[-1] - y_history[-2])
            if y_velocity >= self._vertical_gate:
                self._swipe_confirm_count[hand_no] = 0
                return Gesture.NONE

        # Gate: if 2+ fingers are curled, hand is forming a fist — suppress swipe
        if self._curled_count.get(hand_no, 0) >= 2:
            self._swipe_confirm_count[hand_no] = 0
            return Gesture.NONE

        # Continuous per-frame velocity with 2-frame confirmation:
        # A swipe only fires after the threshold is exceeded for
        # _swipe_confirm_needed consecutive frames, filtering jitter.
        if prev_wrist_x is not None:
            velocity = hand_wrist_x - prev_wrist_x
            is_right = hand.handedness == "Right"

            candidate = Gesture.NONE
            if is_right and velocity >= self._swipe_velocity_threshold:
                candidate = Gesture.SWIPE_LEFT
            elif not is_right and velocity <= -self._swipe_velocity_threshold:
                candidate = Gesture.SWIPE_RIGHT

            if candidate is not Gesture.NONE:
                count = self._swipe_confirm_count.get(hand_no, 0) + 1
                self._swipe_confirm_count[hand_no] = count
                if count >= self._swipe_confirm_needed:
                    self._swipe_confirm_count[hand_no] = 0
                    self._swipe_drop_cooldown[hand_no] = self._swipe_drop_cooldown_frames
                    return candidate
                return Gesture.NONE
            else:
                # Reset confirmation counter when velocity drops below threshold
                self._swipe_confirm_count[hand_no] = 0

            return Gesture.NONE

        return Gesture.NONE

    def _detect_drop(self, hand: HandData, hand_no: int,
                     hand_wrist_x: float, hand_wrist_y: float,
                     prev_wrist_x: float | None) -> Gesture:
        # Initialize per-hand state on first sighting
        if hand_no not in self._y_history:
            self._y_history[hand_no] = deque(maxlen=2)
            self._prev_y[hand_no] = None
            self._hard_drop_cooldown[hand_no] = 0
            self._swipe_drop_cooldown[hand_no] = 0

        average_y = hand_wrist_y
        average_x = hand_wrist_x
        history = self._y_history[hand_no]

        # Hard drop cooldown — completely blocks all drop detection
        if self._hard_drop_cooldown[hand_no] > 0:
            self._hard_drop_cooldown[hand_no] -= 1
            history.append(average_y)
            self._prev_y[hand_no] = average_y
            return Gesture.NONE

        # Hard drop check — BEFORE the horizontal gate so fast deliberate
        # downward flicks are never suppressed by mild sideways wrist drift
        if self._prev_y[hand_no] is not None:
            velocity = average_y - self._prev_y[hand_no]

            if velocity >= self.hard_drop_velocity:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                self._hard_drop_cooldown[hand_no] = self._hard_drop_cooldown_frames
                self._swipe_drop_cooldown[hand_no] = 0
                return Gesture.SWIPE_DOWN_FAST

        # Gate: if hand is moving sideways with significant speed (swiping),
        # suppress soft drops to avoid accidental wrist-extension drops
        if prev_wrist_x is not None:
            x_velocity = abs(average_x - prev_wrist_x)
            if x_velocity >= self._horizontal_gate:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                return Gesture.NONE

        if self._prev_y[hand_no] is not None:
            # Swipe-drop cooldown — suppress soft drops briefly after a swipe
            if self._swipe_drop_cooldown[hand_no] > 0:
                self._swipe_drop_cooldown[hand_no] -= 1
                self._prev_y[hand_no] = average_y
                history.append(average_y)
                return Gesture.NONE

            # Gate: if 2+ fingers are curled, hand is forming a fist — suppress soft drop
            if self._curled_count.get(hand_no, 0) >= 2:
                self._prev_y[hand_no] = average_y
                history.append(average_y)
                return Gesture.NONE

            # Soft drop — continuous per-frame velocity
            velocity = average_y - self._prev_y[hand_no]
            if velocity >= self._soft_drop_velocity_threshold:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                return Gesture.SWIPE_DOWN_SLOW

        self._prev_y[hand_no] = average_y
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
