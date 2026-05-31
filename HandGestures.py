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
        fist_curled_threshold: float = 0.07,
        min_curled_fingers: int = 4,
        swipe_right_threshold: float = 0.08,
        swipe_left_threshold: float = 0.08,
        hard_drop_velocity: float = 0.07,
        soft_drop_threshold: float = 0.05,
        drop_history_frames: int = 6,
        hard_drop_cooldown: int = 15,
    ) -> None:
        self.fist_curled_threshold = fist_curled_threshold
        self.min_curled_fingers = min_curled_fingers
        self.swipe_right_threshold = swipe_right_threshold
        self.swipe_left_threshold = swipe_left_threshold
        self.hard_drop_velocity = hard_drop_velocity
        self.soft_drop_threshold = soft_drop_threshold

        self._drop_history_frames = drop_history_frames
        self._y_history: deque[float] = deque(maxlen=drop_history_frames)
        self._prev_y: float | None = None
        self._hard_drop_cooldown: int = 0
        self._hard_drop_cooldown_frames: int = hard_drop_cooldown

    """Looks for detection based on current frame"""
    def update(self, hands: list[HandData], hand_no: int = 0) -> GestureState:
        if hand_no >= len(hands):
            self.reset()
            return GestureState(actions=actions_for_gesture(Gesture.NONE))

        #Picks Hand
        hand = hands[hand_no]

        gesture = self._detect_fist(hand)
        if gesture is Gesture.NONE:
            gesture = self._detect_drop(hand)
        if gesture is Gesture.NONE:
            gesture = self._detect_swipe(hand)

        return GestureState(
            gesture=gesture,
            actions=actions_for_gesture(gesture),
        )

    def reset(self) -> None:
        self._y_history.clear()
        self._prev_y = None
        self._hard_drop_cooldown = 0

    @staticmethod
    def _delta_over_window(history: deque[float]) -> float | None:
        """Return newest - oldest once the deque is full; otherwise None."""
        if len(history) < history.maxlen:
            return None
        return history[-1] - history[0]

    def _detect_fist(self, hand: HandData) -> Gesture:
        curled_count = 0

        #either hand, a curled finger has its tip close to its mcp
        for tip_idx, mcp_idx in _FIST_TIP_MCP_PAIRS:
            #[x, y, z] based on finger landmark
            tip = hand.landmarks_norm[tip_idx]
            mcp = hand.landmarks_norm[mcp_idx]
            #calculate its distances based on x and y,
            distance = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1], tip[2] - mcp[2]) 

            if distance < self.fist_curled_threshold:
                curled_count += 1

        if curled_count >= self.min_curled_fingers:
            return Gesture.FIST
        return Gesture.NONE

    def _detect_swipe(self, hand: HandData) -> Gesture:
        return Gesture.NONE

    #more akin to a flick of the wrist vs slowly droping now.
    def _detect_drop(self, hand: HandData) -> Gesture:
        average_y = sum(lm[1] for lm in hand.landmarks_norm) / len(hand.landmarks_norm)

        # Cooldown
        if self._hard_drop_cooldown > 0:
            self._hard_drop_cooldown -= 1
            self._y_history.append(average_y)
            self._prev_y = average_y
            return Gesture.NONE

        #Hard drop - Instant
        if self._prev_y is not None:
            velocity = average_y - self._prev_y
            if velocity >= self.hard_drop_velocity:
                self._y_history.append(average_y)
                self._prev_y = average_y
                self._hard_drop_cooldown = self._hard_drop_cooldown_frames
                return Gesture.SWIPE_DOWN_FAST

        self._prev_y = average_y

        #Soft drop - Sustained
        self._y_history.append(average_y)
        delta = self._delta_over_window(self._y_history)
        if delta is None:
            return Gesture.NONE
        if delta >= self.soft_drop_threshold:
            return Gesture.SWIPE_DOWN_SLOW
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
