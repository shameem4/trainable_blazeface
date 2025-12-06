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

Shared utilities to eliminate code duplication across the codebase:

| File | Description |
|------|-------------|
| `model_utils.py` | Model loading (`load_model()`), device setup (`setup_device()`) |
| `drawing.py` | Visualization (`draw_detections()`, `draw_ground_truth_boxes()`, `draw_fps()`, `draw_info_text()`) |
| `metrics.py` | Evaluation metrics (`compute_iou()`, `match_detections_to_ground_truth()`) |
| `video_utils.py` | Video capture (`WebcamVideoStream`, `FPSCounter`) |
| `augmentation.py` | Image augmentation (saturation, brightness, flip, occlusion) |
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

**Resuming Training**: Pass `--resume <checkpoint_path>` to continue from a
previously saved checkpoint. Leave `--resume` unset to start from scratch (or
from the MediaPipe weights if `--init-weights mediapipe`).

### Trainer Configuration Knobs

`train_blazeface.py` exposes several CLI switches to rebalance the classifier and regression heads:

- `--use-focal-loss / --no-focal-loss`: Focal loss is now the default. Disable it if you need classic BCE behaviour.
- `--positive-classification-weight FLOAT`: Multiplies only the positive classification term (defaults to 80.0) so high-quality positives dominate the top-k scoring process.
- `--hard-negative-ratio FLOAT`: Number of negatives mined per positive (defaults to 1.5). Raising the ratio emphasises background suppression; lowering it emphasises positives.
- `--detection-weight` / `--classification-weight`: Maintain the classic BlazeFace weighting of 150 / 40 unless experimenting.
- `--freeze-thaw` + `--freeze-epochs` / `--unfreeze-mid-epochs`: Enable the staged warm-up (defaults 2 and 3 epochs respectively) so only the detection heads train first, then `backbone2`, before full fine-tuning. Set the durations to `0` if you want to skip a phase.
- Metrics such as **Pos Acc** now evaluate at a 0.45 decision threshold and only consider detections with scores ≥0.10 when computing mAP—mirroring the thresholds used for inference.

These levers were introduced while debugging the score ordering issues highlighted in the [Medium BlazeFace article](#references); make sure to log their values (the trainer prints them at startup) whenever sharing results.

### Anchor System

The model uses 896 anchors distributed across two feature map scales:

- **16×16 grid**: 2 anchors per cell (512 anchors)
- **8×8 grid**: 6 anchors per cell (384 anchors)

Anchor format: `[x_center, y_center, width, height]`

- Default: `width = height = 1.0` (fixed anchor size)
- Configurable for variable anchor sizes if needed

### Loss Function

Based on vincent1bt's implementation with additional tunable knobs:

- **Classification**: Focal loss (default) or BCE with hard negative mining
- **Positive emphasis**: Separate positive classification weight (`--positive-classification-weight`, default 80.0) so foreground logits can outrank background anchors
- **Regression**: Huber loss for box coordinates
- **Negative mining ratio**: 1.5:1 (negatives to positives) via `--hard-negative-ratio`
- **Loss weights**: detection=150.0, classification background=40.0, classification positive configurable (default 80.0)

These settings align the repository with lessons from production BlazeFace deployments (see [Resources](#references)) and make it easier to diagnose scoring mismatches between ground truth and predictions.

### Training Data Formats

**CSV Format (default)**:

- Use `split_dataset.py` to create `train.csv` / `val.csv`
- CSV rows follow WIDER Face style (`image_path, x1, y1, w, h, ...`)
- Point `--train-data` / `--val-data` at those CSVs

## Debugging & Diagnostics

Use `debug_training.py` for single-image, end-to-end inspection of the preprocessing, anchor assignment, and score-ranking pipeline:

- **Anchor unit tests**: Synthetic boxes ensure anchors are assigned and decoded correctly.
- **CSV encode/decode test**: Re-encodes positives from the dataset and checks `decode_boxes()` reconstructs them with IoU 1.0.
- **Scoring diagnostics**: Prints mean positive score, score/IoU correlation, and counts of positives in the top-k anchors—mirroring the troubleshooting workflow recommended in [the BlazeFace Medium article](#references).
- **Visualization overlays**: Saves side-by-side GT vs. prediction images for manual review.

Run it without arguments to sample 10 random rows, or use `--index 42` (or a comma-separated list) to inspect specific samples.

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

### **Training Methodology Alignment**

Despite architectural differences, training follows vincent1bt's validated approach:

- Same loss formulation and weights
- Identical hard negative mining strategy
- Similar augmentation pipeline
- ~500 epoch convergence expectation

This ensures training effectiveness while gaining PyTorch/MediaPipe ecosystem benefits.

### Comparison with vincent-vdb/medium_posts BlazeFace

[vincent-vdb's implementation](https://github.com/vincent-vdb/medium_posts/tree/main/blazeface/python) provides a clean PyTorch reference. Here are the key differences:


#### **Target Encoding Strategy**

**vincent-vdb**: Bipartite matching assigns each GT to best prior, then encodes offsets
**Our Implementation**: Each GT finds best anchor via IoU, multiple GTs can match same anchor location

**Impact**: Our approach simpler but may struggle with overlapping faces; vincent-vdb ensures unique GT-prior pairing.

#### Why These Deviations?

1. **MediaPipe Alignment**: Our implementation prioritizes compatibility with MediaPipe pretrained weights
2. **Simplicity**: Detection-only model (no classification scores) reduces complexity
3. **Research Focus**: More modular structure for experimentation vs. vincent-vdb's production focus
4. **Dataset Flexibility**: Support multiple annotation formats vs. vincent-vdb's YOLO-specific pipeline

Both implementations are valid BlazeFace interpretations with different design priorities: vincent-vdb optimizes for TFLite deployment, while ours prioritizes MediaPipe compatibility and research flexibility.

## References

- [MediaPipe BlazeFace](https://google.github.io/mediapipe/solutions/face_detection.html) - Original BlazeFace paper and implementation
- [vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow) - TensorFlow training implementation
- [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) - PyTorch implementation and model conversion
- [zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/) - PyTorch MediaPipe models
- [vincent-vdb/medium_posts](https://github.com/vincent-vdb/medium_posts/tree/main/blazeface/python) - Lightweight PyTorch reference used for comparison
- [BlazeFace Medium article](https://medium.com/data-science/blazeface-how-to-run-real-time-object-detection-in-the-browser-66c2ac9acd75) - Practical notes on debugging score ordering and deploying BlazeFace in browsers

### Current State Analysis

The codebase (6,359 lines of Python) is functionally sound but has technical debt that impacts maintainability:

#### Critical Issues

**1. Code Duplication** (High Priority)
- **3 IoU implementations** with inconsistent behavior:
  - `blazebase.py:69-95` - NumPy with +1 offset
  - `dataloader.py:23-34` - NumPy with +1 offset
  - `metrics.py:12-38` - NumPy without offset (DIFFERENT!)
- **2 box encoding functions** with diverging logic
- ~180 lines of duplicate code


**3. Limited Test Coverage** (Medium Priority)
- Only **2.6% coverage** (165 test lines / 6,359 code lines)
- Missing tests for model forward pass, loss computation, inference pipeline

#### Phase 2: Code Deduplication (Week 1-2)

Create canonical implementations in `core/` package:

```python
# core/geometry.py
def compute_iou_np(box1: np.ndarray, box2: np.ndarray, offset: bool = True) -> np.ndarray:
    """Unified NumPy IoU implementation"""
    # Single canonical implementation

def compute_iou_torch(box1: torch.Tensor, box2: torch.Tensor, offset: bool = True) -> torch.Tensor:
    """Unified PyTorch IoU implementation"""
    # Single canonical implementation

# core/anchor_encoding.py
def generate_anchors(config: ModelConfig) -> np.ndarray:
    """Unified anchor generation"""

def encode_boxes_to_anchors_np(boxes: np.ndarray, anchors: np.ndarray) -> tuple:
    """Unified NumPy encoding"""
```

#### Phase 4: Test Coverage Expansion (Week 3-4)

Target >65% coverage:

```python
# tests/test_geometry.py
def test_iou_perfect_overlap()
def test_iou_no_overlap()
def test_iou_torch_matches_numpy()

# tests/test_blazeface.py
def test_forward_shape()
def test_load_weights()

# tests/test_losses.py
def test_loss_computation()
def test_hard_negative_mining()
```


#### Step 3: Create Unified IoU (1 hour)

```python
# core/geometry.py
import numpy as np
import torch

def compute_iou_np(box1: np.ndarray, box2: np.ndarray,
                   offset: bool = True) -> np.ndarray:
    """
    Compute IoU between two sets of boxes using NumPy.

    Args:
        box1: Boxes in [y1, x1, y2, x2] format, shape (N, 4)
        box2: Boxes in [y1, x1, y2, x2] format, shape (M, 4)
        offset: If True, adds +1 to width/height (MediaPipe compatibility)

    Returns:
        IoU matrix of shape (N, M)
    """
    # Ensure 2D arrays
    if box1.ndim == 1:
        box1 = box1.reshape(1, -1)
    if box2.ndim == 1:
        box2 = box2.reshape(1, -1)

    # Extract coordinates
    y1_1, x1_1, y2_1, x2_1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    y1_2, x1_2, y2_2, x2_2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Compute intersection
    y1_inter = np.maximum(y1_1[:, None], y1_2[None, :])
    x1_inter = np.maximum(x1_1[:, None], x1_2[None, :])
    y2_inter = np.minimum(y2_1[:, None], y2_2[None, :])
    x2_inter = np.minimum(x2_1[:, None], x2_2[None, :])

    inter_h = np.maximum(0, y2_inter - y1_inter)
    inter_w = np.maximum(0, x2_inter - x1_inter)

    if offset:
        inter_area = (inter_h + 1) * (inter_w + 1)
    else:
        inter_area = inter_h * inter_w

    # Compute union
    if offset:
        area1 = (y2_1 - y1_1 + 1) * (x2_1 - x1_1 + 1)
        area2 = (y2_2 - y1_2 + 1) * (x2_2 - x1_2 + 1)
    else:
        area1 = (y2_1 - y1_1) * (x2_1 - x1_1)
        area2 = (y2_2 - y1_2) * (x2_2 - x1_2)

    union_area = area1[:, None] + area2[None, :] - inter_area

    # Compute IoU
    iou = inter_area / np.maximum(union_area, 1e-8)
    return iou

def compute_iou_torch(box1: torch.Tensor, box2: torch.Tensor,
                      offset: bool = True) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes using PyTorch.

    Args:
        box1: Boxes in [y1, x1, y2, x2] format, shape (N, 4)
        box2: Boxes in [y1, x1, y2, x2] format, shape (M, 4)
        offset: If True, adds +1 to width/height

    Returns:
        IoU matrix of shape (N, M)
    """
    # Ensure 2D tensors
    if box1.ndim == 1:
        box1 = box1.unsqueeze(0)
    if box2.ndim == 1:
        box2 = box2.unsqueeze(0)

    # Extract coordinates
    y1_1, x1_1, y2_1, x2_1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    y1_2, x1_2, y2_2, x2_2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Compute intersection
    y1_inter = torch.maximum(y1_1[:, None], y1_2[None, :])
    x1_inter = torch.maximum(x1_1[:, None], x1_2[None, :])
    y2_inter = torch.minimum(y2_1[:, None], y2_2[None, :])
    x2_inter = torch.minimum(x2_1[:, None], x2_2[None, :])

    inter_h = torch.clamp(y2_inter - y1_inter, min=0)
    inter_w = torch.clamp(x2_inter - x1_inter, min=0)

    if offset:
        inter_area = (inter_h + 1) * (inter_w + 1)
    else:
        inter_area = inter_h * inter_w

    # Compute union
    if offset:
        area1 = (y2_1 - y1_1 + 1) * (x2_1 - x1_1 + 1)
        area2 = (y2_2 - y1_2 + 1) * (x2_2 - x1_2 + 1)
    else:
        area1 = (y2_1 - y1_1) * (x2_1 - x1_1)
        area2 = (y2_2 - y1_2) * (x2_2 - x1_2)

    union_area = area1[:, None] + area2[None, :] - inter_area

    # Compute IoU
    iou = inter_area / torch.clamp(union_area, min=1e-8)
    return iou
```

#### Step 4: Update Files to Remove Duplicates

#### Step 5: Write Tests

```python
# tests/test_geometry.py
import numpy as np
import torch
import pytest
from core.geometry import compute_iou_np, compute_iou_torch

def test_iou_np_perfect_overlap():
    """Test IoU with identical boxes (should be 1.0)"""
    box = np.array([[0, 0, 10, 10]])
    iou = compute_iou_np(box, box)
    assert np.abs(iou[0, 0] - 1.0) < 1e-6

def test_iou_np_no_overlap():
    """Test IoU with non-overlapping boxes (should be 0.0)"""
    box1 = np.array([[0, 0, 10, 10]])
    box2 = np.array([[20, 20, 30, 30]])
    iou = compute_iou_np(box1, box2)
    assert iou[0, 0] == 0.0

def test_iou_torch_matches_numpy():
    """Ensure PyTorch and NumPy versions match"""
    box1_np = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
    box2_np = np.array([[2, 2, 12, 12]])

    box1_torch = torch.from_numpy(box1_np).float()
    box2_torch = torch.from_numpy(box2_np).float()

    iou_np = compute_iou_np(box1_np, box2_np)
    iou_torch = compute_iou_torch(box1_torch, box2_torch).numpy()

    assert np.allclose(iou_np, iou_torch, atol=1e-6)
```

```
earmesh/
├── config/              # Centralized configuration
├── core/                # Core utilities (zero duplication)
├── models/              # Model definitions
├── data/                # Data pipeline
├── training/            # Training components
├── inference/           # Inference utilities
└── utils/               # General utilities
```


## License

See individual attribution sources for their respective licenses.
