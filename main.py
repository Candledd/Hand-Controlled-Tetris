from __future__ import annotations

import os
import sys
import contextlib

# Silence TensorFlow, MediaPipe, and other log noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

@contextlib.contextmanager
def suppress_stderr():
    """Redirect both Python sys.stderr and underlying C++ FD 2 to devnull."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        stderr_fd = None

    if stderr_fd is not None:
        try:
            dup_stderr = os.dup(stderr_fd)
        except OSError:
            dup_stderr = None
    else:
        dup_stderr = None

    with open(os.devnull, 'wb') as devnull:
        old_stderr = sys.stderr
        # Redirect Python's stderr
        sys.stderr = devnull
        # Redirect C++ FD level stderr
        if stderr_fd is not None:
            try:
                os.dup2(devnull.fileno(), stderr_fd)
            except OSError:
                pass
        try:
            yield
        finally:
            sys.stderr = old_stderr
            if dup_stderr is not None:
                try:
                    os.dup2(dup_stderr, stderr_fd)
                    os.close(dup_stderr)
                except OSError:
                    pass

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(os.path.join(script_dir, "tetris"))

with suppress_stderr():
    import cv2
    from tetris.HandTrackingModule import HandTracker
    from tetris.HandGestures import GestureDetector
    from tetris.GestureConversions import Win32Keyboard
    from tetris.GameLogic import TetrisKeyboardDispatcher

# Shows camera info
SHOW_FEEDBACK_WINDOW = True

def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}", file=sys.stderr)
        sys.exit(1)

    keyboard = Win32Keyboard()
    dispatcher = TetrisKeyboardDispatcher(keyboard)
    model_path = os.path.join(script_dir, "hand_landmarker.task")

    # Suppress verbose MediaPipe XNNPACK/TFLite delegate logs on initialization
    with suppress_stderr():
        tracker = HandTracker(max_hands=2, model_path=model_path)

    last_action = None

    try:
        with tracker:
            with GestureDetector() as detector:
                while True:
                    success, img = cap.read()
                    if not success:
                        break

                    img, hands = tracker.find_hands(img, draw=SHOW_FEEDBACK_WINDOW)
                    if SHOW_FEEDBACK_WINDOW:
                        img = tracker.label_hands(img)

                    state = detector.update(hands)
                    dispatcher.dispatch(state)

                    if SHOW_FEEDBACK_WINDOW:
                        detector.draw_gesture_overlay(img, state)
                        dispatcher.draw_pressed_keys_overlay(img, dispatcher, keyboard)

                    cv2.imshow("Hand Gestures", img)

                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        dispatcher.release_all()
        keyboard.shutdown()
        cap.release()
        if SHOW_FEEDBACK_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Exit cleanly on Ctrl+C without showing traceback
        sys.exit(0)
