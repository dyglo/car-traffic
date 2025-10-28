# Car Traffic Counter

[![Python badge](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics badge](https://img.shields.io/badge/YOLOv8-ultralytics-17a2b8?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![OpenCV badge](https://img.shields.io/badge/OpenCV-vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

Real-time vehicle counting for multi-lane highways powered by Ultralytics YOLOv8 detection and SORT tracking. The application overlays segmented inbound/outbound trigger lines, counts direction-specific traffic, and writes an annotated video for dashboards or analytics pipelines.

---

## Table of Contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage Tips](#usage-tips)
7. [Project Structure](#project-structure)
8. [Development Notes](#development-notes)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features
- **Dual-Lane Counting:** Distinct inbound (green) and outbound (red) trigger segments tuned for asymmetric carriageways.
- **Accurate Tracking:** Combines YOLOv8 detections with SORT to follow vehicles through occlusion and re-identify lanes.
- **Live HUD Overlay:** Displays total, inbound, and outbound counts with custom dashboard graphics.
- **Mask Support:** Optional binary mask suppresses detections outside the area of interest.
- **Exportable Results:** Annotated footage is written to `result.mp4` for later review.

## Architecture
- 🎯 **Detection:** Ultralytics YOLOv8 (`yolov8l.pt`) identifies cars, buses, trucks, and motorbikes.
- 🧭 **Tracking:** SORT handles ID assignment and trajectory smoothing.
- 🪄 **Business Logic:** Horizontal line segments per direction validate crossings before incrementing counters.
- 🖼️ **Rendering:** OpenCV + CVZone draw bounding boxes, count widgets, and the trigger segments.

## Prerequisites
- Python 3.10 or newer.
- GPU optional (CPU works for recorded footage, GPU recommended for live feeds).
- [Ultralytics YOLOv8 weights](https://docs.ultralytics.com/models/yolov8/) — download `yolov8l.pt` manually and keep it outside version control.

### Core Python Packages
- 🧠 `ultralytics` – YOLOv8 detection
- 👁️ `opencv-python` – video I/O & drawing
- 🧮 `numpy` – matrix operations
- 🪪 `cvzone` – HUD widgets
- 🔁 `filterpy` (via `requirements.txt`) – SORT dependencies

Install them with `pip install -r requirements.txt` after cloning.

## Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/dyglo/car-traffic.git
   cd car-traffic
   ```
2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Add YOLO weights**
   - Download `yolov8l.pt` from Ultralytics.
   - Place it in the project root (same folder as `main.py`) or update the path in `main.py`.
   - Do **not** commit model weights to Git.
5. **Run the app**
   ```bash
   python main.py
   ```
6. **Review outputs**
   - The annotated window opens during processing (press `q` to exit).
   - Processed footage saves to `result.mp4` in the project root.

## Configuration
Key parameters live near the top of `main.py`:

| Setting | Description |
| --- | --- |
| `LINE_IN_PERCENT` / `LINE_OUT_PERCENT` | Vertical placement of inbound/outbound trigger segments (percentage of frame height). |
| `LINE_IN_X_RANGE` / `LINE_OUT_X_RANGE` | Horizontal coverage (fractions of frame width) to keep counts lane-specific. |
| `LINE_SEGMENT_PADDING` | Horizontal tolerance ensuring wide vehicles trigger correctly. |
| `Sort` constructor args | `max_age`, `min_hits`, `iou_threshold` tune tracker sensitivity. |
| `mask.png` | Optional ROI mask (white = keep, black = ignore). |

Edit these values to match new camera angles. Re-run `python main.py` after each tweak to validate alignment.

## Usage Tips
- Keep the camera fixed; moving cameras require additional stabilization or homography adjustments.
- When swapping input video, update `assets/traffic.mp4` or pass a new path to `cv.VideoCapture`.
- If detections stutter, try a lighter model (`yolov8m.pt`) or enable GPU acceleration via CUDA.
- For live dashboards, remove or adapt the `cv.imshow` call and stream frames to your UI layer.

## Project Structure
```
.
├─ assets/
│  ├─ traffic.mp4          # Sample footage (replace with your feed)
│  ├─ graphics.png         # Total-count HUD asset
│  ├─ graphics1.png        # In/Out indicator HUD asset
│  └─ mask.png             # ROI mask (black = ignored)
├─ main.py                 # Application entry point
├─ sort.py                 # SORT tracker implementation
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation
└─ CONTRIBUTIONS.md        # Community workflow guidelines
```

## Development Notes
- `result.mp4` and any ad-hoc data exports are ignored by Git via `.gitignore`.
- Model weights (`*.pt`, `*.onnx`, etc.) must stay out of version control.
- Use `python -m py_compile main.py` or `flake8`/`black` (optional) to lint before opening pull requests.

## Contributing
We welcome improvements! See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for branching strategy, coding standards, and pull-request guidelines. Forks are encouraged—just remember to host large assets (weights, long recordings) in external storage.

## License
This project is released under the MIT License. Add a `LICENSE` file to your fork if you plan to redistribute modified versions.
