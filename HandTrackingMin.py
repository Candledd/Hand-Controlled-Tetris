import cv2
import mediapipe as mp
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

def draw_landmarks_on_video(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if not detection_result or not detection_result.hand_landmarks:
        return annotated_image

    #for each hand
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        #draws the circles and lines
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, img = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms=int(time.time() * 1000))

            annotated_image = draw_landmarks_on_video(img_rgb, detection_result)
            cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
