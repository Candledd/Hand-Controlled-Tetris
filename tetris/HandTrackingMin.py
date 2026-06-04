import cv2
import time

try:
    from tetris.HandTrackingModule import HandTracker
except ModuleNotFoundError:
    from HandTrackingModule import HandTracker


def draw_fps(img, fps: float) -> None:
    cv2.putText(
        img,
        str(int(fps)),
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
    )


def main() -> None:
    cap = cv2.VideoCapture(0)
    pTime = 0

    with HandTracker(max_hands=2) as tracker:
        while True:
            success, img = cap.read()
            if not success:
                break

            img, hands = tracker.find_hands(img)

            if hands:
                index_tip = tracker.find_position(hand_no=0, landmark_id=8)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
