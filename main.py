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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

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
                    success, img = cap.read()
                    if not success:
                        break

                    img, hands = tracker.find_hands(img, draw=not args.no_preview)
                    if not args.no_preview:
                        img = tracker.label_hands(img)

                    state = detector.update(hands)
                    dispatcher.dispatch(state)

                    if not args.no_preview:
                        detector.draw_gesture_overlay(img, state)
                        dispatcher.draw_pressed_keys_overlay(img, dispatcher, keyboard)
                        cv2.imshow("Hand Gestures", img)
                        
                        # Break loop if the window is closed or ESC key is pressed
                        if cv2.getWindowProperty("Hand Gestures", cv2.WND_PROP_VISIBLE) < 1:
                            break
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
    finally:
        dispatcher.release_all()
        keyboard.shutdown()
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Exit cleanly on Ctrl+C without showing traceback
        sys.exit(0)
