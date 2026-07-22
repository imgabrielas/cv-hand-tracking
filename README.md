# cv-hand-tracking

Real-time hand tracking desktop app for macOS, built with OpenCV and MediaPipe. Captures your Mac's built-in webcam, detects up to two hands, and overlays their landmarks and connections live in a mirrored, resizable window.

## Features

- Live hand landmark detection and skeleton overlay (MediaPipe Hands), with distinct colors for the first and second detected hand
- Bounding box drawn around each detected hand
- Mirrored view (raise your right hand, see it on the right)
- FPS counter
- On-screen "Quit" button, plus `Q` / `Esc` keyboard shortcuts to close the window
- Automatically selects the Mac's built-in camera over a connected iPhone (Continuity Camera), even when the iPhone would otherwise take priority
- Custom UI styling (SF Compact font, rounded button, configurable color palette) rendered via Pillow

## Requirements

- macOS (uses AVFoundation for camera selection)
- Python 3.11
- A webcam

## Setup

```bash
python3.11 -m venv mp_env
mp_env/bin/pip install -r requirements.txt
```

## Usage

```bash
mp_env/bin/python main.py
```

A window opens showing your webcam feed with hand tracking overlaid. Close it via the on-screen "Quit" button, or by pressing `Q` or `Esc`.

## Notes

- `requirements.txt` pins `mediapipe==0.10.21` deliberately: newer `mediapipe` releases (0.10.30+) dropped the legacy `mediapipe.solutions` API this app relies on for macOS builds.
- Camera selection, colors, frame size, and font are configured via constants near the top of `main.py`.
