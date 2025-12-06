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

**Recent Refactoring:**

- Eliminated 200+ lines of duplicated code across demo scripts and data loaders
- Fixed critical anchor size bug in `dataloader.py` (was using 1/32, 1/16 instead of 1/16, 1/8)
- Removed duplicate `compute_iou_batch()` function from `loss_functions.py`
- Consolidated augmentation code into `utils/augmentation.py`
- Removed unused `decode_for_loss()` method from `blazedetector.py`
- Synced the inference pipeline with [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch), including keypoint decoding and denormalization fixes for ROI extraction

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
python split_dataset.py --csv data/raw/blazeface/fixed_images.csv --output-dir data/splits

# 2. Train with MediaPipe initialization (default, recommended)
python train_blazeface.py \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --epochs 500

# 3. Or train from scratch (random initialization)
python train_blazeface.py \
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

## Architecture Comparison

### Comparison with vincent1bt/blazeface-tensorflow

This implementation follows [vincent1bt's training methodology](https://github.com/vincent1bt/blazeface-tensorflow) with several architectural deviations:

#### Identical Elements (vincent1bt)

✅ **Anchor Configuration**: 16×16 grid (512 anchors) + 8×8 grid (384 anchors) = 896 total
✅ **Loss Weights**: Detection=150.0, Classification=35.0
✅ **Hard Negative Mining**: 3:1 ratio (negatives to positives)
✅ **Augmentation**: Random saturation (0.5-1.5), brightness (±0.2), horizontal flip (50% each)
✅ **Input Size**: 128×128 RGB images
✅ **Loss Functions**: Huber loss for regression, binary cross-entropy for classification

#### Key Deviations (vincent1bt)

#### 1. **Anchor Format Difference**

| vincent1bt | Our Implementation |
|------------|-------------------|
| `[class, x1, y1, x2, y2]` (corners) | `[class, ymin, xmin, ymax, xmax]` (MediaPipe convention) |
| Anchors stored as corner coordinates | Anchors stored with center + size encoding |

**Impact**: Box encoding/decoding logic differs but mathematically equivalent.

#### 2. **Framework & Model Structure**

| vincent1bt | Our Implementation |
|------------|-------------------|
| TensorFlow 2.0 | PyTorch |
| Standard convolutions | BlazeBlock architecture (depthwise separable) |
| Single model file | Modular: `BlazeDetector` base + `BlazeFace` model |

**Impact**: Our implementation closer to MediaPipe's mobile-optimized architecture.

#### 3. **Data Pipeline**

| vincent1bt | Our Implementation |
|------------|-------------------|
| tf.data.Dataset generators | PyTorch Dataset + DataLoader |
| WIDER FACE + FDDB datasets | Configurable (CSV/NPY formats) |
| Batch-first processing | Flexible batch handling |

**Impact**: More flexible data source support, easier to integrate custom datasets.

#### 4. **Weight Loading**

| vincent1bt | Our Implementation |
|------------|-------------------|
| TensorFlow SavedModel format | MediaPipe `.pth` + training `.ckpt` |
| N/A | Automatic BlazeBlock_WT → BlazeBlock conversion |

**Impact**: Can load MediaPipe pretrained weights and fine-tune on custom data.

#### 5. **Additional Features (Not in vincent1bt)**

- **Landmark Support**: `blazeface_landmark.py` for 468-point face landmarks
- **Multiple Annotation Formats**: COCO JSON, CSV, PTS support via `decoder.py`
- **Demo Scripts**: Real-time webcam and dataset browser tools
- **IoU-Based Evaluation**: `image_demo.py` with detection-GT matching
- **Utilities Module**: Shared code for visualization, metrics, augmentation

#### 6. **Anchor Size Bug Fix**

**vincent1bt**: Uses consistent anchor sizes throughout
**Our Implementation**: Had inconsistent anchor sizes between `blazebase.py` (1/16, 1/8) and `dataloader.py` (1/32, 1/16)
**Status**: ✅ Fixed in recent refactoring (now uses 1/16, 1/8 consistently)

### **Why These Deviations?**

1. **MediaPipe Compatibility**: Enable loading official MediaPipe weights for transfer learning
2. **PyTorch Ecosystem**: Leverage PyTorch's research-friendly API and tooling
3. **Modularity**: Separate detector/landmarker for cleaner architecture
4. **Flexibility**: Support multiple datasets and annotation formats
5. **Extensibility**: Easy to adapt for non-face detection tasks (e.g., ear detection)

### **Training Methodology Alignment**

Despite architectural differences, training follows vincent1bt's validated approach:

- Same loss formulation and weights
- Identical hard negative mining strategy
- Similar augmentation pipeline
- ~500 epoch convergence expectation

This ensures training effectiveness while gaining PyTorch/MediaPipe ecosystem benefits.

### Comparison with vincent-vdb/medium_posts BlazeFace

[vincent-vdb's implementation](https://github.com/vincent-vdb/medium_posts/tree/main/blazeface/python) provides a clean PyTorch reference. Here are the key differences:

#### Identical Elements (vincent-vdb)

✅ **Framework**: Both use PyTorch
✅ **Anchor Count**: 896 total anchors (loaded from `anchors.npy` in vincent-vdb)
✅ **BlazeBlock Architecture**: Depthwise separable convolutions
✅ **Scale Factors**: 128.0 for front model (our implementation)
✅ **Multi-scale Detection**: 8×8 and 16×16 feature pyramid

#### Key Deviations (vincent-vdb)

| Aspect | vincent-vdb | Our Implementation |
|--------|-------------|-------------------|
| **Anchor Storage** | Pre-computed `anchors.npy` file | Generated at runtime via `generate_reference_anchors()` |
| **Model Variants** | Front (128×128) + Back (256×256) | Front only (128×128) |
| **Output Format** | `[batch, 896, 7]` (4 box + 3 class) | `[batch, 896, 4]` box only (detection-only model) |
| **NMS** | Uses `torchvision.ops.nms()` | Custom weighted NMS in `_weighted_non_max_suppression()` |
| **Box Encoding** | Center format `(cx, cy, w, h)` | MediaPipe format `[ymin, xmin, ymax, xmax]` |
| **Loss Function** | `MultiBoxLoss` (localization + classification) | Custom `BlazeFaceDetectionLoss` with hard negative mining |
| **Target Matching** | `match()` with jaccard/IoU threshold 0.5 | `encode_boxes_to_anchors()` with best IoU anchor |
| **Optimizer** | Adam with ReduceLROnPlateau | AdamW with configurable scheduler |
| **Dataset** | YOLO format from Open Images | CSV/NPY formats (WIDER Face compatible) |
| **Augmentation** | None in provided code | Saturation, brightness, flip, occlusion (via `utils/augmentation.py`) |

#### **Additional Features in Our Implementation**

- **MediaPipe Weight Loading**: Automatic conversion from `BlazeBlock_WT` format
- **Landmark Model**: Separate `blazeface_landmark.py` for 468 keypoints
- **Multiple Annotation Decoders**: COCO, CSV, PTS support
- **Demo Scripts**: Webcam and dataset browser with IoU evaluation
- **Modular Utilities**: Shared visualization, metrics, augmentation modules

#### **Additional Features in vincent-vdb**

- **TensorFlow Lite Export**: `tf_lite_converter.py` for mobile deployment
- **Dual Model Support**: Both front (mobile) and back (desktop) variants
- **Visualization**: Validation grid with GT (green) and predictions (red)
- **Jupyter Notebook**: Interactive training experimentation

#### Box Decoding Comparison

**vincent-vdb**:

```python
# Center-based decoding
cx = prior_cx + variance[0] * delta_cx * prior_w
cy = prior_cy + variance[1] * delta_cy * prior_h
w = prior_w * exp(variance[2] * delta_w)
h = prior_h * exp(variance[3] * delta_h)
```

**Our Implementation**:

```python
# MediaPipe offset decoding
x_center = anchor_x + (pred_x / x_scale)
y_center = anchor_y + (pred_y / y_scale)
w = pred_w / w_scale
h = pred_h / h_scale
```

**Impact**: Different but equivalent parameterizations; vincent-vdb uses variance scaling, we use direct scale factors.

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

## License

See individual attribution sources for their respective licenses.
