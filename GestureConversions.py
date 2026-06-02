import cv2
import time

from HandGestures import GestureDetector
from HandTrackingModule import HandTracker

def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    try:
        with HandTracker(max_hands=2) as tracker:
            with GestureDetector() as detector:
                while True:
                    success, img = cap.read()
                    if not success:
                        break

                    img, hands = tracker.find_hands(img)
                    detector.update(hands)

                    cv2.imshow("Hand Gestures", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()