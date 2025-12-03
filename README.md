# EarMesh - Ear Detection and Landmark Estimation

PyTorch implementation of BlazeFace-style ear detection and landmark estimation models.

## Attribution

This project builds upon the work of:

- **[vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow)**
  \- Training methodology and loss functions for BlazeFace
- **[hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)**
  \- PyTorch BlazeFace implementation and model conversion
- **[zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/)**
  \- PyTorch implementation of MediaPipe models

## Overview

EarMesh adapts Google's BlazeFace architecture (originally designed for face
detection) to detect and extract landmarks from human ears. The project
provides:

- **Ear Detection**: Lightweight BlazeFace-style detector for bounding boxes
- **Ear Landmarks**: Keypoint estimation for ear anatomical features
- **Training Pipeline**: Complete training infrastructure following vincent1bt

## Architecture

The detection model follows the BlazeFace architecture:

- **Input**: 128×128 RGB images
- **Output**: 896 anchor predictions with bounding boxes and scores
- **Anchor Format**: `[x_center, y_center, width, height]` configurable
- **Box Format**: `[ymin, xmin, ymax, xmax]` (MediaPipe convention)

### Model Variants

| Variant | Description | Output |
|---------|-------------|--------|
| `BlazeBlock` | Trainable with explicit BatchNorm | 4 coords (box) |
| `BlazeBlock_WT` | MediaPipe with folded BatchNorm | 16 coords (box+kpts) |

The trainable model (`BlazeBlock`) outputs only bounding boxes (4 coordinates
per anchor), while the original MediaPipe format (`BlazeBlock_WT`) includes 6
keypoints (16 coordinates total).

### Weight Conversion

When loading MediaPipe pretrained weights (`.pth`), the
`load_mediapipe_weights()` function:

1. Converts backbone from `BlazeBlock_WT` (folded BatchNorm) to `BlazeBlock`
   (explicit BatchNorm)
2. Extracts box-only regressor weights (first 4 of 16 coords per anchor),
   discarding keypoint channels
3. Copies classifier weights directly (unchanged)

## Module Structure

### Core Modules

| File | Description |
|------|-------------|
| `blazebase.py` | Base classes, anchor generation, weight conversion |
| `blazedetector.py` | Base detector with preprocessing, NMS, decoding |
| `blazelandmarker.py` | Base landmark with ROI extraction, denormalization |
| `blazeface.py` | BlazeFace detection model implementation |
| `blazeface_landmark.py` | Face landmark model (468 points) implementation |
| `decoder.py` | Annotation format decoders (COCO, CSV, PTS) |

### Utility Modules (`utils/`)

Shared utilities for demo scripts to eliminate code duplication:

| File | Description |
|------|-------------|
| `model_utils.py` | Model loading (`load_model()`), device setup (`setup_device()`) |
| `drawing.py` | Visualization (`draw_detections()`, `draw_ground_truth_boxes()`, `draw_fps()`, `draw_info_text()`) |
| `metrics.py` | Evaluation metrics (`compute_iou()`, `match_detections_to_ground_truth()`) |
| `video_utils.py` | Video capture (`WebcamVideoStream`, `FPSCounter`) |
| `config.py` | Configuration constants (paths, thresholds, anchor settings) |

### Demo Scripts

| File | Description |
|------|-------------|
| `webcam_demo.py` | Real-time detection from webcam |
| `image_demo.py` | Image-by-image detection with ground truth comparison |

## Model Weights

Place weight files in the `model_weights/` directory:

- `blazeface.pth` - Pre-trained MediaPipe BlazeFace weights
- `blazeface_landmark.pth` - Pre-trained face landmark weights
- `*.ckpt` - Custom trained checkpoint files (from training pipeline)

### Loading Weights

| Weight Type | Format | Conversion |
|-------------|--------|------------|
| MediaPipe (`.pth`) | `BlazeBlock_WT` 16 coords | To `BlazeBlock` 4 coords |
| Checkpoint (`.ckpt`) | `BlazeBlock` with 4 coords | Loaded directly |

## Usage

### Basic Detection with MediaPipe Weights

```python
import torch
from blazeface import BlazeFace
from blazebase import anchor_options, load_mediapipe_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detector
detector = BlazeFace().to(device)

# Load MediaPipe weights (auto-converts BlazeBlock_WT -> BlazeBlock)
load_mediapipe_weights(detector, "model_weights/blazeface.pth")
detector.eval()
detector.generate_anchors(anchor_options)

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

### Running Demos

**Webcam Demo** - Real-time detection:

```bash
# Default MediaPipe weights
python webcam_demo.py

# Custom weights with threshold
python webcam_demo.py --weights checkpoints/BlazeFace_best.pth --threshold 0.5

# Disable mirror mode
python webcam_demo.py --no-mirror
```

**Image Demo** - Browse dataset with ground truth comparison:

```bash
# Detection only (default)
python image_demo.py

# Comparison mode with ground truth and IoU matching
python image_demo.py --no-detection-only

# Custom weights and CSV
python image_demo.py --weights checkpoints/custom.pth --csv data/splits/val.csv --threshold 0.3
```

Controls for image demo:

- `A` / `Left Arrow` - Previous image
- `D` / `Right Arrow` - Next image
- `Q` / `ESC` - Quit

Both demos automatically detect weight format (`.pth` for MediaPipe, `.ckpt` for custom trained).

## Detection Output Format

Each detection is a tensor of 5 values:

| Index | Description |
|-------|-------------|
| `[0:4]` | Bounding box: `ymin, xmin, ymax, xmax` (normalized 0-1) |
| `[4]` | Confidence score |

## Training

Training follows the methodology from [vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow):

### Quick Start

```bash
# 1. Split your CSV dataset into train/val
python csv_dataloader.py --csv data/raw/blazeface/fixed_images.csv --output data/splits

# 2. Train with MediaPipe initialization (default, recommended)
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --epochs 500

# 3. Or train from scratch (random initialization)
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --init-weights scratch
```

### Weight Initialization

The trainer supports two initialization strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `mediapipe` (default) | Load MediaPipe pretrained weights via `load_mediapipe_weights()` | Transfer learning |
| `scratch` | Random initialization | Training from scratch |

**MediaPipe Initialization** (default):

- Uses `load_mediapipe_weights()` to load and convert BlazeBlock_WT weights
- Automatically handles BatchNorm unfolding
- Extracts box-only weights (4 coords) from face model (16 coords)
- Recommended for faster convergence

**Auto-Resume**: The trainer automatically resumes from
`checkpoints/BlazeFace_best.pth` if it exists. Use `--no-auto-resume` to start
fresh with MediaPipe/scratch weights, or `--resume <path>` to load a specific
checkpoint.

### Anchor System

The model uses 896 anchors distributed across two feature map scales:

- **16×16 grid**: 2 anchors per cell (512 anchors)
- **8×8 grid**: 6 anchors per cell (384 anchors)

Anchor format: `[x_center, y_center, width, height]`

- Default: `width = height = 1.0` (fixed anchor size)
- Configurable for variable anchor sizes if needed

### Loss Function

Based on vincent1bt's implementation:

- **Classification**: Binary cross-entropy (or focal loss) with hard negative mining
- **Regression**: Huber loss for box coordinates
- **Negative mining ratio**: 3:1 (negatives to positives)
- **Loss weights**: detection=150.0, classification=35.0

### Training Data Formats

**CSV Format** (recommended):

- See `csv_dataloader.py` for CSV-based training
- Supports WIDER Face format with `image_path, x1, y1, w, h, width, height`
- Use `--csv-format` flag with training script

**NPY Format** (legacy):

- Preprocessed data stored in `data/preprocessed/`
- `train_detector.npy`, `val_detector.npy` - Detection training data
- `train_landmarker.npy`, `val_landmarker.npy` - Landmark training data

Raw annotations are in `data/raw/` in various formats (COCO, CSV, PTS).

## Annotation Formats

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
