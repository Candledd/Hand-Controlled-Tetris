# Tetris Hand Gestures Controller

This project enables you to play Tetris using computer vision and hand gestures. It uses the computer webcam to track your hands via Google MediaPipe and translates specific gestures into synthetic keyboard events (using the Win32 API) that control the game.

## Features & Controls

The script detects gestures and simulates corresponding keyboard inputs. The current mappings are:

*   **Move Left**: Swipe left with your hand. (Sends `Left Arrow` / `Numpad 4`)
*   **Move Right**: Swipe right with your hand. (Sends `Right Arrow` / `Numpad 6`)
*   **Rotate Clockwise**: Close your **Right Hand** into a fist. (Sends `Up Arrow` / `Numpad 9`)
*   **Rotate Counterclockwise**: Close your **Left Hand** into a fist. (Sends `Ctrl` / `Z`)
*   **Soft Drop**: Swipe down slowly. (Sends `Down Arrow` / `Numpad 2`)
*   **Hard Drop**: Swipe down quickly. (Sends `Space` / `Numpad 8`)
*   **Hold Piece**: Make a "Peace" sign (Index + Middle fingers extended, others curled). (Sends `Shift` / `C`)
*   **Retry / Restart**: Make a "Rock 'n' Roll" sign (Index + Pinky fingers extended, others curled). (Sends `R`)

## Prerequisites

- Python 3.9+
- A working webcam
- Windows OS (Required for Win32 synthetic keyboard inputs)

## Setup & Installation

1. **Clone or Download** the repository to your local machine.
2. **Create a virtual environment** (recommended):
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install the dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   *(This will install `opencv-python`, `mediapipe`, and `pytest`)*

## Running the Application

1. Make sure your virtual environment is active.
2. Run the main orchestrator script:
   ```cmd
   python main.py
   ```
3. A window titled "Hand Gestures" will pop up, displaying your webcam feed with gesture and action overlays.
4. Open your Tetris game and ensure it is the active/focused window.
5. The gesture system will now send keyboard commands to your active window!

## Toggling Keyboard Synthesis

If you need to quickly pause the gesture inputs (for example, to type somewhere else):
- Press **`Ctrl + Alt + G`** at any time to suspend/resume keyboard synthesis.

## Troubleshooting
- **No camera feed?** If your camera doesn't open, you may need to adjust the `camera_index` passed into `main(camera_index=0)` inside `main.py` (e.g. change `0` to `1`).
- **Gestures too sensitive / false positives?** You can tweak the gesture parameters inside the `GestureDetector` constructor in `tetris/HandGestures.py`. 
