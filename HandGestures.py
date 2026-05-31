"""
All hand gesture logic lives here.

To add a gesture:
  1. Add a value to `Gesture`.
  2. Map it in `GESTURE_ACTIONS`.
  3. Implement a `_detect_*` method on `GestureDetector`.
  4. Call that method from `GestureDetector.update()`.

Run directly to preview on the webcam:
    python HandGestures.py
"""

from __future__ import annotations

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


@dataclass
class GestureState:
    gesture: Gesture = Gesture.NONE
    actions: dict[str, bool] = field(default_factory=dict)

    @property
    def active_actions(self) -> list[str]:
        return [name for name, pressed in self.actions.items() if pressed]


def actions_for_gesture(gesture: Gesture) -> dict[str, bool]:
    actions = {name: False for name in ALL_ACTIONS}
    action = GESTURE_ACTIONS.get(gesture)
    if action:
        actions[action] = True
    return actions


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

"""Converts `HandData` lists into `GestureState` each frame."""
class GestureDetector:
    def __init__(self, fist_curled_threshold: float = 0.08, min_curled_fingers: int = 4) -> None:
        self.fist_curled_threshold = fist_curled_threshold
        self.min_curled_fingers = min_curled_fingers

    def update(self, hands: list[HandData], hand_no: int = 0) -> GestureState:
        if hand_no >= len(hands):
            self.reset()
            return GestureState(actions=actions_for_gesture(Gesture.NONE))

        hand = hands[hand_no]

        gesture = self._detect_fist(hand)

        # Future swipe detectors:
#        if gesture is Gesture.NONE:
#            gesture = self._detect_swipe(hand)

        return GestureState(
            gesture=gesture,
            actions=actions_for_gesture(gesture),
        )

    def reset(self) -> None:
        pass

    def _detect_fist(self, hand: HandData) -> Gesture:
        norm = hand.landmarks_norm
        curled_count = 0

        for tip_idx, mcp_idx in _FIST_TIP_MCP_PAIRS:
            tip = norm[tip_idx]
            mcp = norm[mcp_idx]
            distance = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])
            if distance < self.fist_curled_threshold:
                curled_count += 1

        if curled_count >= self.min_curled_fingers:
            return Gesture.FIST
        return Gesture.NONE

    def _detect_swipe_left(self, hand: HandData) -> Gesture:
        return Gesture.NONE









def main() -> None:
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = GestureDetector()

    with HandTracker(max_hands=1) as tracker:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, hands = tracker.find_hands(img)
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
