# EarMesh

Machine learning pipeline for ear detection and landmark detection on ear images.

## Project Structure

```text
earmesh/
├── common/                         # Common utilities
│   └── image_annotation_viewer.py  # Interactive annotation viewer
├── ear_detector/                   # Ear detection module
│   └── data/raw/                   # Raw detection datasets (COCO, CSV formats)
├── ear_landmarker/                 # Ear landmark detection module
│   └── data/raw/                   # Raw landmark datasets (PTS format)
├── mediapipe/                      # MediaPipe model implementations
│   ├── BlazeFace/                  # BlazeFace face detection model
│   └── facelandmarks/              # Facial landmark model
├── models/                         # Trained model storage
│   ├── ear_detector/
│   └── ear_landmarker/
└── shared/                         # Shared utilities
    ├── data_decoder/               # Annotation format decoders
    └── image_processing/           # Image drawing utilities
```

## Modules

### Shared Utilities

#### Data Decoder (`shared/data_decoder/`)

Handles multiple annotation formats:

**`decoder.py`** - Unified decoder interface

- `find_annotation(image_path)` - Auto-detects and returns annotation
  file path and type
- `decode_annotation(annotation_path, image_path, type)` - Decodes
  annotations to standard format
- `get_annotation_color(type)` - Returns visualization color for
  annotation type

**Format-specific decoders:**

- `coco_decoder.py` - COCO JSON format (bboxes + keypoints)
- `csv_decoder.py` - CSV format (supports both `xmin,ymin,xmax,ymax` and
  `x,y,w,h` bbox formats)
- `pts_decoder.py` - PTS format (keypoints only)

All decoders return standardized format:

```python
[
    {
        'bbox': [x, y, width, height],        # Optional
        'keypoints': [x1, y1, v1, x2, y2, v2, ...]
        # Optional, v=visibility (0-2)
    }
]
```

#### Image Processing (`shared/image_processing/`)

**`annotation_drawer.py`** - Visualization utilities

- `draw_bounding_boxes(draw, annotations, color, width)` - Draw bboxes
  on ImageDraw object
- `draw_keypoints(draw, annotations, color, radius)` - Draw keypoints on
  ImageDraw object
- `draw_annotations_on_image(image_path, annotations, ...)` - Returns
  PIL Image with annotations
- `visualize_annotations(image_path, annotations, ...)` - Draws and
  displays with matplotlib
- `display_image(image, figsize)` - Display PIL Image

### Common Tools

**`image_annotation_viewer.py`** - Interactive annotation viewer

- GUI file picker for selecting images
- Auto-detects annotation format (COCO/CSV/PTS)
- Visualizes bboxes and keypoints with color-coding:
  - Red: COCO annotations
  - Green: CSV annotations
  - Purple: PTS annotations
  - Blue: Keypoints (all formats)

Usage:

```bash
python common/image_annotation_viewer.py
```

### MediaPipe Models

PyTorch implementations of MediaPipe models:

- **BlazeFace** (`mediapipe/BlazeFace/blazeface.py`) - Face detection
  model
- **Facial Landmarks** (`mediapipe/facelandmarks/facial_lm_model.py`) -
  Facial landmark detection

## Data Formats

### Supported Annotation Formats

**COCO JSON** (`.coco.json` suffix)

- Full COCO format with images and annotations
- Supports bounding boxes and keypoints
- Typically named `*_annotations.coco.json`

**CSV** (`_annotations.csv` suffix)

- Column formats supported:
  - `image_path, xmin, ymin, xmax, ymax`
  - `filename, x, y, w, h`
- Multiple annotations per image supported

**PTS** (`.pts` extension)

- Keypoint format:

  ```text
  version: 1
  n_points: N
  {
  x1 y1
  x2 y2
  ...
  }
  ```

- One PTS file per image (same basename)

### Dataset Organization

**Ear Detection** (`ear_detector/data/raw/`)

- Multiple COCO datasets for ear bounding box detection
- CSV annotations from OpenImages
- Train/test/validation splits

**Ear Landmarks** (`ear_landmarker/data/raw/`)

- PTS format keypoint annotations (collectionA, collectionB)
- COCO format with keypoints
- 55 keypoints per ear

## Data Processing

### Preprocessing Raw Data

The `shared/data_processing/data_processor.py` script converts raw
annotations into preprocessed NPZ files for training.

**Features:**

- Parallel processing with multiprocessing
- Memory-efficient batch processing with periodic disk flushing
- Automatic format detection (COCO, CSV, PTS)
- Error logging and propagation
- Temporary file caching to reduce memory footprint

**Usage:**

```bash
# Process all datasets (detector, landmarker, teacher)
python shared/data_processing/data_processor.py --all

# Process specific datasets only
python shared/data_processing/data_processor.py --detector
python shared/data_processing/data_processor.py --landmarker --teacher

# Custom configuration
python shared/data_processing/data_processor.py --all \
  --batch-size 1000 --workers 8 --split 0.85
```

**Arguments:**

- `--all` - Process all datasets
- `--detector` - Process detector data (ear bounding boxes)
- `--landmarker` - Process landmarker data (ear keypoints)
- `--teacher` - Process teacher data (cropped ears for autoencoder)
- `--batch-size N` - Images per batch (default: 500)
- `--workers N` - Parallel workers (default: CPU count)
- `--split RATIO` - Train/val split (default: 0.8)
- `--flush-every N` - Flush to disk every N batches (default: 5)
- `--input-dir PATH` - Raw data directory (default: data/raw)
- `--output-dir PATH` - Output directory (default: data/preprocessed)

**Output Files:**

- `train_detector.npz`, `val_detector.npz` - Ear detection data
- `train_landmarker.npz`, `val_landmarker.npz` - Landmark data
- `train_teacher.npz`, `val_teacher.npz` - Cropped ear images

## Development

### Adding New Annotation Formats

1. Create decoder in `shared/data_decoder/<format>_decoder.py`:

   ```python
   def find_<format>_annotation(image_path):
       # Return annotation file path or None

   def decode_<format>_annotation(annotation_path, image_filename):
       # Return list of {'bbox': [...], 'keypoints': [...]} dicts
   ```

2. Update `shared/data_decoder/decoder.py`:

   - Import new decoder functions
   - Add to `find_annotation()` check sequence
   - Add to `decode_annotation()` switch
   - Add color to `get_annotation_color()`

### Using Utilities in Scripts

```python
# Decode annotations
from shared.data_decoder.decoder import find_annotation, decode_annotation

annotation_path, annotation_type = find_annotation('path/to/image.jpg')
annotations = decode_annotation(annotation_path, 'path/to/image.jpg', annotation_type)

# Visualize
from shared.image_processing.annotation_drawer import visualize_annotations

visualize_annotations('path/to/image.jpg', annotations,
                     bbox_color='red', keypoint_color='blue')
```

## Requirements

- Python 3.x
- PyTorch
- PIL/Pillow
- matplotlib
- numpy
- tkinter (for GUI file picker)
