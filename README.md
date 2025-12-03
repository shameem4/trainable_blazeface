# EarMesh - Ear Detection and Landmark Estimation

PyTorch implementation of BlazeFace-style ear detection and landmark estimation models.

## Attribution

This project builds upon the work of:

- **[vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow)** - Training methodology and loss functions for BlazeFace
- **[hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)** - PyTorch BlazeFace implementation and model conversion
- **[zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/)** - PyTorch implementation of MediaPipe models

## Overview

EarMesh adapts Google's BlazeFace architecture (originally designed for face detection) to detect and extract landmarks from human ears. The project provides:

- **Ear Detection**: Lightweight BlazeFace-style detector for ear bounding boxes
- **Ear Landmarks**: Keypoint estimation for ear anatomical features
- **Training Pipeline**: Complete training infrastructure following vincent1bt's methodology

## Architecture

The detection model follows the BlazeFace architecture:

- **Input**: 128×128 RGB images
- **Output**: 896 anchor predictions with bounding boxes, keypoints, and confidence scores
- **Anchor Format**: `[x_center, y_center, width, height]` with configurable fixed or variable sizes
- **Box Format**: `[ymin, xmin, ymax, xmax]` (MediaPipe convention)

## Module Structure

| File | Description |
|------|-------------|
| `blazebase.py` | Base classes (`BlazeBase`, `BlazeBlock`, `FinalBlazeBlock`), anchor generation, weight conversion |
| `blazedetector.py` | Base detector class with preprocessing, NMS, and anchor decoding |
| `blazelandmarker.py` | Base landmark class with ROI extraction and denormalization |
| `blazeface.py` | BlazeFace detection model implementation |
| `blazeface_landmark.py` | Face landmark model (468 points) implementation |
| `webcam_demo.py` | Demo script for real-time detection |
| `decoder.py` | Annotation format decoders (COCO, CSV, PTS) |

## Model Weights

Place weight files in the `model_weights/` directory:

- `blazeface.pth` - Pre-trained MediaPipe BlazeFace weights
- `blazeface_landmark.pth` - Pre-trained face landmark weights
- `*.ckpt` - Custom trained checkpoint files (from training pipeline)

## Usage

### Basic Detection with MediaPipe Weights

```python
import torch
from blazeface import BlazeFace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detector
detector = BlazeFace().to(device)
detector.load_weights("model_weights/blazeface.pth")

# Run detection on an image (H, W, 3) numpy array
detections = detector.process(image)
```

### Loading Custom Trained Models

```python
import torch
from blazeface import BlazeFace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detector
detector = BlazeFace().to(device)

# Load from training checkpoint
checkpoint = torch.load("model_weights/ear_detector.ckpt")
detector.load_state_dict(checkpoint["model_state_dict"])
detector.eval()

detections = detector.process(image)
```

### Webcam Demo

Run real-time detection demo:

```bash
python webcam_demo.py
```

The demo automatically detects weight format (`.pth` for MediaPipe, `.ckpt` for custom trained).

Press `q` or `Esc` to exit.

## Detection Output Format

Each detection is a tensor of 17 values:

| Index | Description |
|-------|-------------|
| `[0:4]` | Bounding box: `ymin, xmin, ymax, xmax` (normalized 0-1) |
| `[4:16]` | 6 keypoints as (x, y) pairs |
| `[16]` | Confidence score |

## Training

Training follows the methodology from [vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow):

### Anchor System

The model uses 896 anchors distributed across two feature map scales:

- **16×16 grid**: 2 anchors per cell (512 anchors)
- **8×8 grid**: 6 anchors per cell (384 anchors)

Anchor format: `[x_center, y_center, width, height]`

- Default: `width = height = 1.0` (fixed anchor size)
- Configurable for variable anchor sizes if needed

### Loss Function

Based on vincent1bt's implementation:

- **Classification**: Binary cross-entropy with hard negative mining
- **Regression**: Smooth L1 loss for box coordinates
- **Negative mining ratio**: 3:1 (negatives to positives)

### Data Preparation

Preprocessed data is stored in `data/preprocessed/`:

- `train_detector.npy`, `val_detector.npy` - Detection training data
- `train_landmarker.npy`, `val_landmarker.npy` - Landmark training data
- `train_teacher.npy`, `val_teacher.npy` - Teacher model data

Raw annotations are in `data/raw/` in various formats (COCO, CSV, PTS).

## Data Formats

### Supported Annotation Formats

**COCO JSON** (`.coco.json` suffix)

- Full COCO format with images and annotations
- Supports bounding boxes and keypoints

**CSV** (`_annotations.csv` suffix)

- Column formats: `image_path, xmin, ymin, xmax, ymax` or `filename, x, y, w, h`
- Multiple annotations per image supported

**PTS** (`.pts` extension)

- Keypoint format for landmarks:

  ```text
  version: 1
  n_points: N
  {
  x1 y1
  x2 y2
  ...
  }
  ```

## Development

### Anchor Configuration

The anchor system supports both fixed and variable anchor sizes:

```python
from blazebase import generate_reference_anchors

# Fixed anchor size (default) - anchors are [x, y, 1.0, 1.0]
anchors, num_anchors = generate_reference_anchors(input_size=128, fixed_anchor_size=True)

# Variable anchor sizes - anchors store actual w/h values
anchors, num_anchors = generate_reference_anchors(input_size=128, fixed_anchor_size=False)
```

### Adding New Annotation Formats

1. Create decoder in `decoder.py`:

   ```python
   def find_<format>_annotation(image_path):
       # Return annotation file path or None

   def decode_<format>_annotation(annotation_path, image_filename):
       # Return list of {'bbox': [...], 'keypoints': [...]} dicts
   ```

2. Update `find_annotation()` and `decode_annotation()` to include the new format.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- OpenCV (`cv2`)
- PIL/Pillow
- matplotlib

## References

- [MediaPipe BlazeFace](https://google.github.io/mediapipe/solutions/face_detection.html) - Original BlazeFace paper and implementation
- [vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow) - TensorFlow training implementation
- [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) - PyTorch implementation and model conversion
- [zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/) - PyTorch MediaPipe models

## License

See individual attribution sources for their respective licenses.
