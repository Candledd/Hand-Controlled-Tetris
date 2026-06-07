from __future__ import annotations

import time

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# Landmark indices for each fingertip (thumb → pinky).
FINGER_TIPS = (4, 8, 12, 16, 20)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

WRIST = 0
HAND_LABEL_COLOR = (88, 205, 54)

TELEMETRY_KEY = ord("t")

_LANDMARK_LABELS: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_DIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_DIP",
    "MIDDLE_TIP",
    "RING_MCP",
    "RING_PIP",
    "RING_DIP",
    "RING_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

_TIP_SET: frozenset[int] = frozenset(FINGER_TIPS)


@dataclass
class HandData:
    """One detected hand in pixel and normalized coordinates."""

    handedness: str
    landmarks_px: list[tuple[int, int]]
    landmarks_norm: list[tuple[float, float, float]]


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    if not detection_result or not detection_result.hand_landmarks:
        return rgb_image

    annotated_image = np.copy(rgb_image)
    for hand_landmarks in detection_result.hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image


def label_hands_on_image(img: np.ndarray, hands: list[HandData]) -> None:
    """Draw Left/Right labels for each detected hand."""
    for hand in hands:
        if not hand.landmarks_px:
            continue

        wrist_x, wrist_y = hand.landmarks_px[WRIST]
        cv2.putText(
            img,
            hand.handedness,
            (wrist_x - 30, wrist_y - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            HAND_LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )


class HandTracker:
    """MediaPipe hand detection, drawing, and landmark access."""

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        presence_confidence: float = 0.7,
        model_path: str = "tetris/hand_landmarker.task",
    ) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._start_time_ms: float = time.monotonic() * 1000
        self._hands: list[HandData] = []

    def find_hands(self, img: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[HandData]]:
        """Detect hands in a BGR frame. Optionally draw landmarks on the frame."""
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(time.monotonic() * 1000 - self._start_time_ms)
        detection_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        self._hands = self._parse_hands(detection_result, width, height)

        if draw:
            annotated_rgb = draw_landmarks_on_image(img_rgb, detection_result)
            if annotated_rgb is not img_rgb:
                img = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        return img, self._hands

    def find_position(
        self,
        hand_no: int = 0,
        landmark_id: int | None = None,
        draw: bool = False,
        img: np.ndarray | None = None,
    ) -> list | None:
        """
        Return landmark positions for the last detected frame.

        - All landmarks: [[id, x, y], ...]
        - One landmark: [id, x, y]
        """
        if hand_no >= len(self._hands):
            return [] if landmark_id is None else None

        hand = self._hands[hand_no]

        if landmark_id is not None:
            if landmark_id >= len(hand.landmarks_px):
                return None
            x, y = hand.landmarks_px[landmark_id]
            if draw and img is not None:
                cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
            return [landmark_id, x, y]

        positions = [[i, x, y] for i, (x, y) in enumerate(hand.landmarks_px)]
        if draw and img is not None:
            for _, x, y in positions:
                cv2.circle(img, (x, y), 5, (255, 0, 255), cv2.FILLED)
        return positions

    @staticmethod
    def print_telemetry_snapshot(hands: list[HandData]) -> None:
        bar = "=" * 92
        print(f"\n{bar}")
        print("FINGER POSITION TELEMETRY SNAPSHOT")
        print(bar)
        if not hands:
            print("(no hands detected)")
            print(bar)
            return
        for h_idx, hand in enumerate(hands):
            print(f"\nHand {h_idx} ({hand.handedness})")
            print("-" * 92)
            print(
                f"{'ID':>2}  {'NAME':<11}  {'PX_X':>5}  {'PX_Y':>5}  "
                f"{'NORM_X':>7}  {'NORM_Y':>7}  {'NORM_Z':>7}"
            )
            for i, (px, norm) in enumerate(
                zip(hand.landmarks_px, hand.landmarks_norm)
            ):
                label = _LANDMARK_LABELS[i]
                nx, ny, nz = norm
                x, y = px
                print(
                    f"{i:>2}  {label:<11}  {x:>5}  {y:>5}  "
                    f"{nx:>+7.3f}  {ny:>+7.3f}  {nz:>+7.3f}"
                )
        print(bar + "\n", flush=True)

    def label_hands(
        self, img: np.ndarray, hands: list[HandData] | None = None
    ) -> np.ndarray:
        """Draw Left/Right labels on the frame. Uses last detected hands if none passed."""
        label_hands_on_image(img, hands if hands is not None else self._hands)
        return img

    def get_finger_tips(self, hand_no: int = 0) -> list[tuple[int, int]]:
        """Pixel positions of the five fingertips for gesture / movement logic."""
        if hand_no >= len(self._hands):
            return []
        hand = self._hands[hand_no]
        return [hand.landmarks_px[index] for index in FINGER_TIPS]

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    @staticmethod
    def _parse_hands(detection_result, width: int, height: int) -> list[HandData]:
        if not detection_result or not detection_result.hand_landmarks:
            return []

        hands: list[HandData] = []
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[idx][0].category_name
            landmarks_px = [
                (int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks
            ]
            landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
            hands.append(
                HandData(
                    handedness=handedness,
                    landmarks_px=landmarks_px,
                    landmarks_norm=landmarks_norm,
                )
            )
        return hands
    
