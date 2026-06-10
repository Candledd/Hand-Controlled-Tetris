from __future__ import annotations

import threading
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


@dataclass
class HandData:
    """One detected hand in pixel and normalized coordinates."""

    handedness: str
    landmarks_px: list[tuple[int, int]]
    landmarks_norm: list[tuple[float, float, float]]


def draw_landmarks_on_image(image: np.ndarray, detection_result) -> np.ndarray:
    """Draw hand landmarks directly onto the provided image (no copy)."""
    if not detection_result or not detection_result.hand_landmarks:
        return image

    for hand_landmarks in detection_result.hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    return image


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
        self._lock = threading.Lock()
        self._latest_result = None
        self._last_timestamp = 0
        self._hands: list[HandData] = []

        def result_callback(result, _output_image, _timestamp_ms):
            with self._lock:
                self._latest_result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            result_callback=result_callback,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def find_hands(self, img: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[HandData]]:
        """Detect hands in a BGR frame. Optionally draw landmarks on the frame."""
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Wall-clock timestamp (strictly increasing) required by LIVE_STREAM mode.
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_timestamp:
            ts = self._last_timestamp + 1
        self._last_timestamp = ts

        self._landmarker.detect_async(mp_image, ts)

        with self._lock:
            detection_result = self._latest_result

        if detection_result:
            self._hands = self._parse_hands(detection_result, width, height)

        if draw:
            draw_landmarks_on_image(img, detection_result)

        return img, self._hands

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
    
