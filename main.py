from __future__ import annotations

import os
import sys
import contextlib
import argparse
import json

# Silence TensorFlow, MediaPipe, and other log noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

import threading
import time

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
    from tetris.GestureConversions import Win32Keyboard, KeyboardKey, ACTION_KEY_MAP
    from tetris.GameLogic import TetrisKeyboardDispatcher

class CameraStream:
    def __init__(self, camera_id: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        self._frame = None
        self._frame_id = 0
        self._new_frame = False
        self._lock = threading.Lock()
        self._stopped = threading.Event()

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self) -> None:
        """Daemon thread: continuously reads frames, resizes, and keeps the newest."""
        while not self._stopped.is_set():
            success, frame = self.cap.read()
            if success:
                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._new_frame = True
            else:
                time.sleep(0.01)

    def read(self):
        """
        *new_frame_available* is True only when a frame has been captured since
        the previous call.  *frame* is the most recent frame (or None on the
        very first call).  *frame_id* increments monotonically.
        """
        with self._lock:
            frame = self._frame
            frame_id = self._frame_id
            new = self._new_frame
            self._new_frame = False
        return new, frame, frame_id

    def release(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=1.0)
        self.cap.release()

def parse_args():
    parser = argparse.ArgumentParser(description="Tetris Hand Gesture Controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-preview", action="store_true", help="Disable the OpenCV preview window")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def build_action_key_map(config_map: dict) -> dict:
    if not config_map:
        return ACTION_KEY_MAP
    new_map = {}
    for action, key_names in config_map.items():
        keys = []
        for name in key_names:
            try:
                keys.append(KeyboardKey[name.upper()])
            except KeyError:
                print(f"Warning: Unknown key {name} for action {action}", file=sys.stderr)
        new_map[action] = tuple(keys)
    return new_map

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    gesture_config = config.get("gestures", {})
    key_config = config.get("keybindings", {})
    
    action_key_map = build_action_key_map(key_config)

    stream = CameraStream(args.camera)

    keyboard = Win32Keyboard()
    dispatcher = TetrisKeyboardDispatcher(keyboard, action_key_map=action_key_map)
    model_path = os.path.join(script_dir, "tetris/hand_landmarker.task")

    if not os.path.isfile(model_path):
        print(f"Error: MediaPipe model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    with suppress_stderr():
        tracker = HandTracker(max_hands=2, model_path=model_path)
    try:
        with tracker:
            with GestureDetector(**gesture_config) as detector:
                while True:
                    new_frame, img, _ = stream.read()
                    if not new_frame:
                        if not args.no_preview:
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        time.sleep(0.005)
                        continue

                    if img is None:
                        continue

                    img, hands = tracker.find_hands(img, draw=not args.no_preview)
                    if not args.no_preview:
                        img = tracker.label_hands(img)

                    state = detector.update(hands)
                    dispatcher.dispatch(state)

                    if not args.no_preview:
                        detector.draw_gesture_overlay(img, state)
                        dispatcher.draw_pressed_keys_overlay(img, dispatcher, keyboard)
                        cv2.imshow("Hand Gestures", img)
                        
                        if cv2.getWindowProperty("Hand Gestures", cv2.WND_PROP_VISIBLE) < 1:
                            break
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
    finally:
        dispatcher.release_all()
        keyboard.shutdown()
        stream.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
