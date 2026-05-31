"""
To add a gesture:
  1. Add a value to `Gesture`.
  2. Map it in `GESTURE_ACTIONS`.
  3. Implement a `_detect_*` method on `GestureDetector`.
  4. Call that method from `GestureDetector.update()`.
"""

from __future__ import annotations
from collections import deque

import math
from dataclasses import dataclass, field
from enum import Enum

import cv2
import time

from HandTrackingModule import HandData, HandTracker

WRIST = 0
MIDDLE_TIP = 12

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


"""Init"""
@dataclass
class GestureState:
    gesture: Gesture = Gesture.NONE
    actions: dict[str, bool] = field(default_factory=dict)

    #filters list into only gestures that == true.
    @property
    def active_actions(self) -> list[str]:
        return [name for name, pressed in self.actions.items() if pressed]

"""Ensures only one gesture is true, turns everything else false"""
def actions_for_gesture(gesture: Gesture) -> dict[str, bool]:
    actions = {name: False for name in ALL_ACTIONS}
    action = GESTURE_ACTIONS.get(gesture)
    if action:
        actions[action] = True
    return actions

"""Draws overlay: Gesture + Action"""
def draw_gesture_overlay(img, state: GestureState) -> None:
    cv2.putText(
        img,
        f"Gesture: {state.gesture.value}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    label = ", ".join(state.active_actions) if state.active_actions else "-"
    cv2.putText(
        img,
        f"Actions: {label}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

"""Converts `HandData` lists into `GestureState` for frame."""
class GestureDetector:
    def __init__(
        self,
        fist_curled_threshold: float = 0.06,
        min_curled_fingers: int = 4,
        hard_drop_velocity: float = 0.06,
        soft_drop_threshold: float = 0.05,
        drop_history_frames: int = 6,
        hard_drop_cooldown: int = 15,
        swipe_threshold: float = 0.08,
        swipe_history_frames: int = 5,
        pitch_forward_cutoff: float = -0.08,
    ) -> None:
        self.fist_curled_threshold = fist_curled_threshold
        self.min_curled_fingers = min_curled_fingers
        self.hard_drop_velocity = hard_drop_velocity
        self.soft_drop_threshold = soft_drop_threshold
        self.swipe_threshold = swipe_threshold
        self._pitch_forward_cutoff = pitch_forward_cutoff

        self._drop_history_frames = drop_history_frames
        self._y_history: dict[int, deque[float]] = {}
        self._prev_y: dict[int, float | None] = {}
        self._prev_x: dict[int, float | None] = {}
        self._hard_drop_cooldown: dict[int, int] = {}
        self._hard_drop_cooldown_frames: int = hard_drop_cooldown
        self._x_history: dict[int, deque[float]] = {}
        self._last_x: dict[int, float | None] = {}
        self._swipe_history_frames: int = swipe_history_frames
        self._swipe_drop_cooldown: dict[int, int] = {}
        self._swipe_drop_cooldown_frames: int = 8
        self._prev_wrist: dict[int, tuple[float, float] | None] = {}
        self._curled_count: dict[int, int] = {}
        self._last_hand_gesture: dict[int, Gesture] = {}
        self._gesture_settle_cooldown: dict[int, int] = {}
        self._gesture_settle_cooldown_frames: int = 3

        # Axis-dominance gates: if the opposite axis exceeds this per-frame
        # velocity, the detector suppresses itself.  Prevents wrist-extension
        # during swipes from triggering drops, and vice versa.
        self._horizontal_gate = swipe_threshold / swipe_history_frames * 1.5
        self._vertical_gate = soft_drop_threshold / drop_history_frames * 2.0

        # Per-frame velocity thresholds for continuous motion
        self._swipe_velocity_threshold = swipe_threshold / swipe_history_frames * 0.5
        self._soft_drop_velocity_threshold = soft_drop_threshold / drop_history_frames * 0.5

    """Looks for detection based on current frame"""
    def update(self, hands: list[HandData]) -> GestureState:
        # Clean up state for hands no longer present
        present = set(range(len(hands)))
        for hand_id in list(self._y_history.keys()):
            if hand_id not in present:
                self._y_history.pop(hand_id, None)
                self._prev_y.pop(hand_id, None)
                self._prev_x.pop(hand_id, None)
                self._hard_drop_cooldown.pop(hand_id, None)
                self._swipe_drop_cooldown.pop(hand_id, None)
        for hand_id in list(self._x_history.keys()):
            if hand_id not in present:
                self._x_history.pop(hand_id, None)
                self._last_x.pop(hand_id, None)
                self._prev_wrist.pop(hand_id, None)
                self._curled_count.pop(hand_id, None)
                self._last_hand_gesture.pop(hand_id, None)
                self._gesture_settle_cooldown.pop(hand_id, None)

        if not hands:
            return GestureState(actions=actions_for_gesture(Gesture.NONE))

        combined_actions = {name: False for name in ALL_ACTIONS}
        last_gesture = Gesture.NONE

        for hand_no, hand in enumerate(hands):
            gesture = self._detect_drop(hand, hand_no)
            if gesture is Gesture.NONE:
                gesture = self._detect_fist(hand, hand_no)
            if gesture is Gesture.NONE:
                gesture = self._detect_swipe(hand, hand_no)

            gesture = self._debounce_gesture(hand_no, gesture)

            if gesture is not Gesture.NONE:
                if last_gesture is Gesture.NONE:
                    last_gesture = gesture
                actions = actions_for_gesture(gesture)
                for name in ALL_ACTIONS:
                    if actions[name]:
                        combined_actions[name] = True

        return GestureState(
            gesture=last_gesture,
            actions=combined_actions,
        )

    def reset(self) -> None:
        self._y_history.clear()
        self._prev_y.clear()
        self._prev_x.clear()
        self._hard_drop_cooldown.clear()
        self._x_history.clear()
        self._last_x.clear()
        self._swipe_drop_cooldown.clear()
        self._prev_wrist.clear()
        self._curled_count.clear()
        self._last_hand_gesture.clear()
        self._gesture_settle_cooldown.clear()

    @staticmethod
    def _delta_over_window(history: deque[float]) -> float | None:
        """Return newest - oldest once the deque is full; otherwise None."""
        if len(history) < history.maxlen:
            return None
        return history[-1] - history[0]

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

    def _detect_fist(self, hand: HandData, hand_no: int) -> Gesture:
        wrist = hand.landmarks_norm[WRIST]
        wrist_x, wrist_y = wrist[0], wrist[1]

        # During motion, require one more curled finger to trigger fist
        min_needed = self.min_curled_fingers
        if hand_no in self._prev_wrist and self._prev_wrist[hand_no] is not None:
            prev_x, prev_y = self._prev_wrist[hand_no]
            velocity = abs(wrist_x - prev_x) + abs(wrist_y - prev_y)
            if velocity >= self._soft_drop_velocity_threshold * 2:
                min_needed += 1

        # When fingers pitch toward the camera (z is negative), require more curled
        # fingers to prevent accidental fist during downward swipes
        if hand.landmarks_norm[MIDDLE_TIP][2] < self._pitch_forward_cutoff:
            min_needed += 1

        self._prev_wrist[hand_no] = (wrist_x, wrist_y)

        curled_count = 0
        for tip_idx, mcp_idx in _FIST_TIP_MCP_PAIRS:
            tip = hand.landmarks_norm[tip_idx]
            mcp = hand.landmarks_norm[mcp_idx]
            distance = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])

            if distance < self.fist_curled_threshold:
                curled_count += 1

        self._curled_count[hand_no] = curled_count

        if curled_count >= min_needed:
            return Gesture.FIST
        return Gesture.NONE

    def _detect_swipe(self, hand: HandData, hand_no: int) -> Gesture:
        # Initialize per-hand state on first sighting
        if hand_no not in self._x_history:
            self._x_history[hand_no] = deque(maxlen=self._swipe_history_frames)
            self._last_x[hand_no] = None

        average_x = sum(lm[0] for lm in hand.landmarks_norm) / len(hand.landmarks_norm)
        history = self._x_history[hand_no]

        # Gate: if the hand is moving downward significantly (dropping),
        # suppress swipe detection (uses y-history from _detect_drop)
        y_history = self._y_history.get(hand_no)
        if y_history is not None and len(y_history) >= 2:
            y_velocity = abs(y_history[-1] - y_history[-2])
            if y_velocity >= self._vertical_gate:
                history.append(average_x)
                self._last_x[hand_no] = average_x
                return Gesture.NONE

        # Gate: if 2+ fingers are curled, hand is forming a fist — suppress swipe
        if self._curled_count.get(hand_no, 0) >= 2:
            history.append(average_x)
            self._last_x[hand_no] = average_x
            return Gesture.NONE

        # Continuous per-frame velocity: fires every frame the hand moves
        # in the allowed direction above threshold (no cooldown)
        if self._last_x[hand_no] is not None:
            velocity = average_x - self._last_x[hand_no]
            is_right = hand.handedness == "Right"

            gesture = Gesture.NONE
            if is_right and velocity >= self._swipe_velocity_threshold:
                gesture = Gesture.SWIPE_LEFT
            elif not is_right and velocity <= -self._swipe_velocity_threshold:
                gesture = Gesture.SWIPE_RIGHT

            if gesture is not Gesture.NONE:
                self._swipe_drop_cooldown[hand_no] = self._swipe_drop_cooldown_frames

            self._last_x[hand_no] = average_x
            history.append(average_x)
            return gesture

        self._last_x[hand_no] = average_x
        history.append(average_x)
        return Gesture.NONE

    def _detect_drop(self, hand: HandData, hand_no: int) -> Gesture:
        # Initialize per-hand state on first sighting
        if hand_no not in self._y_history:
            self._y_history[hand_no] = deque(maxlen=self._drop_history_frames)
            self._prev_y[hand_no] = None
            self._prev_x[hand_no] = None
            self._hard_drop_cooldown[hand_no] = 0
            self._swipe_drop_cooldown[hand_no] = 0

        average_y = sum(lm[1] for lm in hand.landmarks_norm) / len(hand.landmarks_norm)
        average_x = sum(lm[0] for lm in hand.landmarks_norm) / len(hand.landmarks_norm)
        history = self._y_history[hand_no]

        # Gate: if hand is moving sideways with significant speed (swiping),
        # suppress all drop detection to avoid accidental wrist-extension drops
        if self._prev_x[hand_no] is not None:
            x_velocity = abs(average_x - self._prev_x[hand_no])
            if x_velocity >= self._horizontal_gate:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                self._prev_x[hand_no] = average_x
                self._swipe_drop_cooldown[hand_no] = self._swipe_drop_cooldown_frames
                return Gesture.NONE

        self._prev_x[hand_no] = average_x

        # Hard drop cooldown — completely blocks all drop detection
        if self._hard_drop_cooldown[hand_no] > 0:
            self._hard_drop_cooldown[hand_no] -= 1
            history.append(average_y)
            self._prev_y[hand_no] = average_y
            return Gesture.NONE

        # Per-frame velocity check
        if self._prev_y[hand_no] is not None:
            velocity = average_y - self._prev_y[hand_no]

            # Hard drop — single frame is enough for a fast flick
            if velocity >= self.hard_drop_velocity:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                self._hard_drop_cooldown[hand_no] = self._hard_drop_cooldown_frames
                self._swipe_drop_cooldown[hand_no] = 0
                return Gesture.SWIPE_DOWN_FAST

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
            if velocity >= self._soft_drop_velocity_threshold:
                history.append(average_y)
                self._prev_y[hand_no] = average_y
                return Gesture.SWIPE_DOWN_SLOW

        self._prev_y[hand_no] = average_y
        history.append(average_y)
        return Gesture.NONE







def main() -> None:
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = GestureDetector()

    with HandTracker(max_hands=2) as tracker:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, hands = tracker.find_hands(img)
            img = tracker.label_hands(img)
            state = detector.update(hands)
            draw_gesture_overlay(img, state)
            

            cTime = time.time()
            fps = 1 / (cTime - pTime) if pTime else 0
            pTime = cTime
            cv2.putText(
                img,
                str(int(fps)),
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            cv2.imshow("Hand Gestures", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
