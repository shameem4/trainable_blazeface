# MediaPipe PyTorch Implementation

PyTorch implementation of MediaPipe's BlazeFace detection and face landmark models.

## Attribution

Based on [MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/) by zmurez.

## Overview

This module provides PyTorch implementations of:

- **BlazeFace**: A lightweight face detection model
- **BlazeFaceLandmark**: A 468-point face landmark model

## Module Structure

| File | Description |
|------|-------------|
| `blazebase.py` | Base classes (`BlazeBase`, `BlazeBlock`, `FinalBlazeBlock`) shared by all models |
| `blazedetector.py` | Base detector class with preprocessing, NMS, and anchor handling |
| `blazelandmarker.py` | Base landmark class with ROI extraction and denormalization |
| `blazeface.py` | BlazeFace face detection model implementation |
| `blazeface_landmark.py` | Face landmark model (468 points) implementation |
| `webcam_demo.py` | Demo script for real-time face detection and landmarks |

## Required Model Weights

Place the following weight files in the `model_weights/` directory:

- `blazeface.pth` - BlazeFace detector weights
- `blazeface_landmark.pth` - Face landmark model weights

## Usage

### Basic Detection

```python
import torch
from common.mediapipe import BlazeFace

# Initialize detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = BlazeFace().to(device)
detector.load_weights("path/to/blazeface.pth")

# Run detection on an image (H, W, 3) numpy array
detections = detector.process(image)
```

### Face Landmarks

```python
import torch
from common.mediapipe import BlazeFace, BlazeFaceLandmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
detector = BlazeFace().to(device)
detector.load_weights("path/to/blazeface.pth")

landmarker = BlazeFaceLandmark().to(device)
landmarker.load_weights("path/to/blazeface_landmark.pth")

# Detect faces and get landmarks
detections = detector.process(image)
landmarks, boxes = landmarker.process(image, detections)
```

### Webcam Demo

Run the demo script for real-time face detection:

```bash
cd common/mediapipe
python webcam_demo.py
```

Press `q` or `Esc` to exit.

## Detection Output Format

Each detection is a tensor of 17 values:

- `[0:4]` - Bounding box: `ymin, xmin, ymax, xmax`
- `[4:16]` - 6 keypoints (x, y pairs)
- `[16]` - Confidence score

## Landmark Output Format

The landmark model outputs 468 face landmarks, each with (x, y, z) coordinates.
