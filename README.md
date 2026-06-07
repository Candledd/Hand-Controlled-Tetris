# Tetris Hand Gesture Controller

A real-time computer vision controller that allows you to play Tetris using hand gestures. It captures webcam input, tracks hand landmarks using Google MediaPipe, and translates gestures into low-level Windows keyboard inputs (Win32 API).

---

## Features

- **Low-Latency Keyboard Emulation**: Uses the Win32 API to inject keystrokes directly into active windows, minimizing input delay.
- **Dual-Hand Control**: Maps separate rotation commands depending on which hand performs a gesture.
- **Gesture Stabilization**: Incorporates 3D depth filtering, palm orientation checks, and multi-frame confirmations to minimize false-positive inputs.
- **Visual Overlay Feed**: Provides an OpenCV preview window showing hand landmark tracking, active gestures, and triggered key presses.
- **Global Control Toggle**: Use the `Ctrl + Alt + G` hotkey to temporarily suspend/resume keyboard inputs at any time.

---

## Gesture Mappings

| Gesture | Action | Keys Emulated | Hand State / Pose |
| :--- | :--- | :--- | :--- |
| **Swipe Left** | Move Left | `Left Arrow` / `Numpad 4` | Quick horizontal movement of either hand to the left |
| **Swipe Right** | Move Right | `Right Arrow` / `Numpad 6` | Quick horizontal movement of either hand to the right |
| **Right Fist** | Rotate Clockwise | `Up Arrow` / `Numpad 9` | Close all fingers of the right hand |
| **Left Fist** | Rotate Counterclockwise | `Ctrl` / `Z` | Close all fingers of the left hand |
| **Slow Swipe Down** | Soft Drop | `Down Arrow` / `Numpad 2` | Controlled vertical movement downward |
| **Fast Swipe Down** | Hard Drop | `Space` / `Numpad 8` | High-velocity movement downward |
| **Peace Sign** | Hold Piece | `Shift` / `C` | Index and middle fingers extended; others curled |
| **Rock 'n' Roll** | Retry / Restart | `R` | Index and pinky fingers extended; others curled |

---

## Prerequisites

- **OS**: Windows (Required for low-level keyboard input injection)
- **Hardware**: USB webcam or integrated camera
- **Software**: Python 3.9 or higher

---

## Installation & Setup

### Running from Source

1. **Clone the repository**:
   ```cmd
   git clone https://github.com/your-username/tetris.git
   cd tetris
   ```

2. **Create and activate a virtual environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   *(Installs OpenCV, MediaPipe, and Pytest)*

4. **Run the controller**:
   ```cmd
   python main.py
   ```

---

## Building a Standalone Executable (`.exe`)

You can bundle the application into a single executable that runs on Windows machines without Python installed.

1. **Install PyInstaller** in your virtual environment:
   ```cmd
   pip install pyinstaller
   ```

2. **Build the release package**:
   ```cmd
   pyinstaller --onefile --noconsole --name HCTetris --add-data "tetris/hand_landmarker.task;tetris" --collect-all mediapipe main.py
   ```

3. **Locate the output**:
   - The standalone executable will be generated at `dist/HCTetris.exe`.
   - You can distribute this `.exe` file directly. The temporary `build/` folder and `HCTetris.spec` file can be deleted.

---

## Usage Instructions

1. Start the controller (`python main.py` or double-click `HCTetris.exe`).
2. An OpenCV preview window titled **"Hand Gestures"** will open. Ensure your hands are clearly visible in the camera frame.
3. Open your Tetris game and ensure it is the active, focused window.
4. If you need to temporarily stop inputs (e.g. to type text elsewhere), press the **`Ctrl + Alt + G`** toggle shortcut.
5. To close the program, press the **`Esc`** key on your keyboard or click the **`X`** close button on the preview window.

---

## Technical Implementation Details

### 3D Distance Formulation
To prevent false fist detections when the hand is tilted toward/away from the camera (which collapses the 2D projected distance between finger tips and MCP joints), the controller uses a weighted 3D Euclidean distance:

```
Distance = sqrt(dx² + dy² + w_z * dz²)
```

A weight factor of `w_z = 2.0` scales the depth coordinate (z) to maintain curl estimation accuracy across different camera angles.

### Palm Normal Gate
To suppress false fist/rotation triggers when the hand is horizontal or inverted, the palm's normal vector is computed via the cross-product of the vector from the wrist to the index MCP (`v_a`) and the vector from the wrist to the pinky MCP (`v_b`):

```
normal_z = v_ax * v_by - v_ay * v_bx
```

Rotations are gated so they only trigger if the normal vector aligns within a 90-degree upright cone relative to the camera plane.

### Multi-Frame Confirmations
To avoid accidental drops or lateral shifts from momentary hand drift or tracking jitter, swipes require a high-velocity movement pattern to be sustained across a minimum of $2$ consecutive frames before sending a key event.

---

## Troubleshooting

* **Camera index issues**: If you have multiple cameras and the app fails to open the webcam feed, edit the index in `main.py` where `cv2.VideoCapture(camera_index)` is instantiated (e.g., change `0` to `1` or `2`).
* **Active focus requirement**: Keystrokes are injected globally via the Windows API, meaning they target whichever application currently has active focus. Ensure your Tetris game is the active window before gesturing.
