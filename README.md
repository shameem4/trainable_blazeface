# Trainable BlazeFace

A PyTorch implementation of BlazeFace that supports fine-tuning from MediaPipe's pretrained weights.

---

## Motivation

BlazeFace is a lightweight face detection model developed by Google for mobile applications. It achieves real-time performance (200+ FPS on mobile GPUs) while maintaining reasonable accuracy for its intended use case: detecting a single frontal face in selfie-style images.

However, the pretrained weights distributed through MediaPipe come as a TensorFlow Lite graph, which cannot be trained, and existing PyTorch ports such as [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch) are inference-only.

This repository adds, on top of those ports:

1. Loading of the MediaPipe weights into a plain PyTorch model whose parameters can be fine-tuned
2. A CSV-based dataset with augmentation and MediaPipe-style anchor target encoding
3. A hard-negative-mining detection loss (focal or BCE on logits, Huber on boxes) and a trainer
4. Evaluation and debugging tools

**Important:** MediaPipe's batch normalization is *folded* into the convolution weights and biases, and this repository keeps it that way. The model has **no BatchNorm layers**; the folded conv weights/biases are fine-tuned directly. (Earlier revisions of this README described an "unfolded BatchNorm" architecture; that code was never used by the model and has been removed in v2.)

---

## Overview

```text
MediaPipe weights (.pth, BN folded into convs)
        │  loaded as-is (regressor heads split into box / keypoint parts)
        ▼
BlazeFace (PyTorch, BlazeBlock_WT, no BN)  ◀── CSV images + boxes ──▶ anchor encoding (896 targets)
        │                                                                   │
        └──────────────────────── detection loss ◀─────────────────────────┘
                                         │
                                  AdamW fine-tuning
```

Because the weights are loaded unchanged, the model reproduces MediaPipe's outputs at initialization.

---

## Architecture

### BlazeFace Detector

BlazeFace is an anchor-based single-shot detector (SSD) designed for mobile inference. The architecture uses depthwise separable convolutions throughout, similar to MobileNet.

```text
                           INPUT IMAGE
                           128 × 128 × 3
                                │
                                ▼
                    ┌───────────────────────┐
                    │    Initial Conv       │
                    │    5×5, stride 2      │
                    │    → 24 channels      │
                    └───────────┬───────────┘
                                │
                    ╔═══════════╧═══════════╗
                    ║    BACKBONE 1         ║
                    ║    11 BlazeBlocks     ║
                    ║    24→28→32→...→88    ║
                    ╚═══════════╤═══════════╝
                                │
                    Feature Map: 16 × 16 × 88
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ╔═══════════════════════╗
          │  Classifier 8   │    ║    BACKBONE 2         ║
          │  (2 anchors)    │    ║    5 BlazeBlocks      ║
          └────────┬────────┘    ║    88→96→96→96→96→96  ║
                   │             ╚═══════════╤═══════════╝
                   │                         │
                   │             Feature Map: 8 × 8 × 96
                   │             ┌───────────┴───────────┐
                   │             ▼                       ▼
                   │   ┌─────────────────┐    ┌─────────────────┐
                   │   │  Classifier 16  │    │  Regressor 16   │
                   │   │  (6 anchors)    │    │  (box + kpts)   │
                   │   └────────┬────────┘    └────────┬────────┘
                   │            │                      │
                   │            ▼                      ▼
          ┌────────┴────────────┴──────────────────────┴────────┐
          │                                                     │
          │              896 ANCHOR PREDICTIONS                 │
          │                                                     │
          │   ┌─────────────────────────────────────────────┐   │
          │   │  512 small anchors (16×16 grid × 2/cell)    │   │
          │   │  384 large anchors (8×8 grid × 6/cell)      │   │
          │   └─────────────────────────────────────────────┘   │
          │                                                     │
          └─────────────────────────────────────────────────────┘
```

### The BlazeBlock

Each block is a depthwise-separable convolution with a skip connection (the conv biases carry the folded BatchNorm):

```text
         Input
           │
     ┌─────┴─────┐
     ▼           │
┌──────────┐     │  skip: max-pool when stride 2, channel zero-pad
│Depthwise │     │  (or 1×1 projection) when channels grow
│ Conv 3×3 │     │
├──────────┤     │
│Pointwise │     │
│ Conv 1×1 │     │
└────┬─────┘     │
     └─────┬─────┘
         (+) Add
           │
         ReLU (PReLU in some blocks)
           │
        Output
```

Depthwise-separable convs cut parameters roughly 9× versus a dense 3×3 conv (`C·9 + C²` vs `9·C²`). The whole model has 101,390 parameters.

---

## Anchor System

BlazeFace predicts face locations relative to a grid of **anchor boxes**. This is the heart of how single-shot detectors work.

### Anchor Grid Visualization

```text
                    16×16 Grid (512 anchors)                    8×8 Grid (384 anchors)
              ┌───────────────────────────────┐           ┌───────────────────────┐
              │ · · · · · · · · · · · · · · · │           │ · · · · · · · · · · ·│
              │ · · · · · · · · · · · · · · · │           │ · · · · · · · · · · ·│
              │ · · ┌─┐ · · · · · · · · · · · │           │ · · ┌───┐ · · · · · ·│
              │ · · │•│ · · · · · · · · · · · │           │ · · │ • │ · · · · · ·│
              │ · · └─┘ · · · · · · · · · · · │           │ · · └───┘ · · · · · ·│
              │ · · · · · · · · · · · · · · · │           │ · · · · · · · · · · ·│
              │ · · · · · · · · · · · · · · · │           │ · · · · · · · · · · ·│
              │ · · · · · · · · · · · · · · · │           │ · · · · · · · · · · ·│
              │ · · · · · · · · · · · · · · · │           └───────────────────────┘
              │ · · · · · · · · · · · · · · · │
              │ · · · · · · · · · · · · · · · │           Each cell: 6 anchors
              │ · · · · · · · · · · · · · · · │           (for larger faces)
              │ · · · · · · · · · · · · · · · │
              │ · · · · · · · · · · · · · · · │
              │ · · · · · · · · · · · · · · · │
              │ · · · · · · · · · · · · · · · │
              └───────────────────────────────┘

              Each cell: 2 anchors
              (for smaller faces)

                        TOTAL: 512 + 384 = 896 anchors
```

### How Anchors Work

For each anchor, the model predicts:

- **Classification score**: "Is there a face here?" (0 to 1)
- **Box regression**: Offset from anchor center to actual face location (Δy, Δx, Δh, Δw)
- **Keypoints**: 6 facial landmarks (eyes, nose, mouth, ears)

```text
Ground Truth Face              Anchor Grid                 Prediction
┌─────────────────┐    ┌─────────────────────────┐    ┌─────────────────┐
│   ┌─────────┐   │    │   ·   ·   ·   ·   ·     │    │  score = 0.95   │
│   │  😊     │   │    │   ·   ·   ·   ·   ·     │    │  Δy = +0.12     │
│   │         │   │    │   ·   ·   •───┐  ·     │    │  Δx = -0.08     │
│   └─────────┘   │────▶│   ·   ·   ·   │   ·     │────▶│  Δh = +0.31     │
│                 │    │   ·   ·   ·   │   ·     │    │  Δw = +0.25     │
│                 │    │   ·   ·   ·   ·   ·     │    │                 │
└─────────────────┘    └─────────────────────────┘    └─────────────────┘
                              ▲
                              │
                        Best matching
                        anchor (highest IoU)
```

---

## Training Pipeline

### Data Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CSV Data                     Image Loading                                │
│   ┌─────────┐                 ┌─────────────┐                               │
│   │image,   │                 │ Load image  │                               │
│   │x1,y1,   │───────────────▶│ Resize/pad  │                               │
│   │w,h      │                 │ to 128×128  │                               │
│   └─────────┘                 └──────┬──────┘                               │
│                                      │                                      │
│                                      ▼                                      │
│                           ┌──────────────────┐                              │
│                           │   Augmentation   │                              │
│                           │ • Flip           │                              │
│                           │ • Brightness     │                              │
│                           │ • Saturation     │                              │
│                           │ • Color jitter   │                              │
│                           │ • Scale          │                              │
│                           │ • Rotation       │                              │
│                           │ • Occlusion      │                              │
│                           │ • Cutout         │                              │
│                           └────────┬─────────┘                              │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│   ┌───────────┐            ┌───────────────┐          ┌────────────┐        │
│   │  Anchor   │            │    Model      │          │   Ground   │        │
│   │  Encoding │            │   Forward     │          │   Truth    │        │
│   │           │            │   Pass        │          │   Targets  │        │
│   │ Box→896   │            │               │          │            │        │
│   │ targets   │            │  BlazeFace()  │          │  cls: 0/1  │        │
│   └─────┬─────┘            └───────┬───────┘          │  box: YXYX │        │
│         │                          │                  └──────┬─────┘        │
│         │                          │                         │              │
│         ▼                          ▼                         ▼              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                           LOSS FUNCTION                         │       │
│   │                                                                 │       │
│   │   L = w_box × L_box + w_bg × L_bg + w_pos × L_pos               │       │
│   │                                                                 │       │
│   │   L_box: Huber loss on positive anchor box predictions          │       │
│   │   L_bg:  Focal loss on hard negative backgrounds                │       │
│   │   L_pos: Focal loss on positive anchors (faces)                 │       │
│   │                                                                 │       │
│   │   Hard Negative Mining: select ratio × positives negatives      │       │
│   │   (highest scoring false positives are most informative)        │       │
│   │                                                                 │       │
│   └───────────────────────────────────┬─────────────────────────────┘       │
│                                       │                                     │
│                                       ▼                                     │
│                              ┌─────────────────┐                            │
│                              │   Backprop +    │                            │
│                              │   AdamW Update  │                            │
│                              └─────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Loss Functions

`train_blazeface.py` defaults: `w_box=200`, `w_bg=60`, `w_pos=80`, hard-negative ratio 2, focal loss (α=0.25, γ=2). The `BlazeFaceDetectionLoss` class itself defaults to 150/35/35, ratio 1, BCE. Classification losses are computed from **raw logits** (`binary_cross_entropy_with_logits`); the earlier probability-space version clamped at 1e-7 and silently dropped the gradient of confidently-wrong anchors.

Object detection training faces significant class imbalance: for every anchor matching a face, there are typically dozens matching background. The loss function addresses this through several mechanisms.

#### Focal Loss (Classification)

Standard cross-entropy treats all examples equally. Focal loss applies a modulating factor that down-weights easy examples (obvious backgrounds) and increases the relative importance of hard cases:

```text
                Standard BCE                    Focal Loss (γ=2)
        │                                │
   Loss │\                               │\
        │ \                              │ \
        │  \                             │  \__________
        │   \                            │
        │    \_____                      │
        │                                │
        └────────────▶                   └────────────▶
           Confidence                       Confidence

"Easy examples dominate             "Hard examples dominate
 gradient updates"                   gradient updates"
```

#### Hard Negative Mining

Rather than using all 800+ background anchors, only the most confident false positives are selected for training. These hard negatives—backgrounds the model incorrectly classifies with high confidence—provide the most informative gradient signal.

#### Huber Loss (Box Regression)

Huber loss combines properties of L1 and L2 losses: quadratic for small errors (smooth gradients near the optimum) and linear for large errors (robustness to outliers):

```text
            L2 Loss                         Huber Loss
        │      /                        │
   Loss │     /                         │     /
        │    /                          │    /
        │   /                           │   ╱ (linear for large errors)
        │  /                            │  ╱
        │ /                             │ ╱_____ (quadratic for small errors)
        └────────────▶                  └────────────▶
           Error                           Error
```

---

## Data Augmentation

Applied after resize/letterbox to 128×128. Each transform is independent: 50% probability for color and geometric transforms, 30% for occlusion-style transforms.

| Augmentation | Probability | Description |
| --- | --- | --- |
| Saturation | 50% | HSV saturation × U(0.5, 1.5) |
| Brightness | 50% | Add U(−0.2, +0.2) × 255 to all channels |
| Color jitter | 50% | Hue ±0.1 × 180, contrast × U(0.8, 1.2) |
| Horizontal flip | 50% | Mirror image and boxes |
| Random scale | 50% | 0.85–1.15×, random crop (zoom in) or black pad (zoom out); boxes < 2% of the frame are dropped |
| Rotation | 50% | ±10°; the new box is the axis-aligned extent of the rotated inscribed ellipse (the corner hull over-inflates round objects) |
| Synthetic occlusion | 30% | Black rectangle 10–50 px |
| Cutout | 30% | Gray (128) square 10–25 px |

Occlusion and cutout are re-drawn (up to 10 tries, then skipped) when they would cover more than 25% of any labelled box, so a positive label never points at an erased object.

---

## Experimental Results

> **Status: the fine-tuned checkpoint in this repo does not beat the frozen MediaPipe weights.** The numbers below were re-measured on 2026-10-03 with the v2 evaluation code against the full validation split (`data/splits/val.csv`, 8,081 images / 20,505 boxes). The previous README's results (500-image subset, "threshold 0.2", per-epoch IoU table, 68% mAP) could not be reproduced: that run used an older trainer, and its evaluation tool silently ignored thresholds below 0.55. Those tables were removed rather than kept as unverifiable claims.

The checkpoint evaluated is `runs/checkpoints/BlazeFace_best.pth` (17 epochs), which was trained with the **pre-v2 code** (probability-space loss with a gradient clamp, 192-image validation slice for checkpoint selection, corner-hull rotation labels). It has not been retrained with v2.

### Detection-level precision / recall (IoU ≥ 0.5, full val split)

Ground truth is the **RetinaFace pseudo-label** CSV (see [Known Limitations](#known-limitations)), not human annotation.

| Model | Score thr | Detections | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| MediaPipe (frozen) | 0.2 | 14,035 | 67.1% | 45.9% | 54.5% |
| MediaPipe (frozen) | 0.5 | 9,556 | 93.8% | 43.7% | 59.6% |
| Fine-tuned (pre-v2 code, epoch 17) | 0.2 | 383,213 | 2.7% | 50.5% | 5.1% |
| Fine-tuned (pre-v2 code, epoch 17) | 0.5 | 48,366 | 19.4% | 45.8% | 27.3% |
| Fine-tuned (pre-v2 code, epoch 17) | 0.75 | 5,452 | 95.2% | 25.3% | 40.0% |

At matched precision (~94–95%) the fine-tuned checkpoint recalls 25% of faces versus 44% for the frozen baseline, i.e. its scores are poorly calibrated and it is worse overall. Reproduce with:

```bash
python utils/debug_training.py --run-eval --weights runs/checkpoints/BlazeFace_best.pth \
    --eval-max-images 100000 --eval-score-threshold 0.5
```

### Training-time metrics (same checkpoint, full val split)

| Metric | Value | Definition |
| --- | --- | --- |
| Val loss | 6.70 | Hard-negative-mined loss (not comparable across loss settings) |
| Positive accuracy | 73.5% | Fraction of positive *anchors* (8 per box) scoring > 0.45 |
| Background accuracy | 96.1% | Over all negative anchors, dominated by easy ones |
| Mean IoU | 0.578 | Decoded box vs. GT on positive anchors (not post-NMS) |
| AP@0.5 | 0.758 | **Per-image** 11-point AP, averaged over images; not COCO/WIDER mAP |

For reference, the first 192 validation images (what the old trainer used to pick "best") gave loss 9.92 and background accuracy 73%, versus 6.70 and 96% on the full set: a small unshuffled slice is not representative.

### Debug Frames

Gray boxes are the ground-truth CSV, green are detections; produced by `utils/debug_training.py` (pre-v2 checkpoint).

| Parade crowd (dense occlusion) | Political rally (high contrast) | Pool deck (specular highlights) |
| :---: | :---: | :---: |
| ![Parade debug frame](assets/latest_run/sample_parade.png) | ![Protest debug frame](assets/latest_run/sample_protest.png) | ![Swimming debug frame](assets/latest_run/sample_swim.png) |

---

## Usage

### Installation

```bash
git clone https://github.com/shameem4/trainable_blazeface.git
cd trainable_blazeface
pip install -r requirements.txt
```

### Training from MediaPipe Weights

```bash
# Fine-tune on your data starting from MediaPipe weights
python train_blazeface.py \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001

# Continue from a checkpoint (never automatic). The LR/cosine schedule follow the
# current --lr/--epochs; only weights, optimizer moments and counters are restored.
python train_blazeface.py --resume runs/checkpoints/BlazeFace_best.pth --epochs 20 ...

# Or train from scratch (random initialization)
python train_blazeface.py \
    --init-weights scratch \
    --train-data data/splits/train.csv \
    ...
```

### Data Format

Prepare a CSV with columns:

```csv
image_path,x1,y1,w,h
path/to/image1.jpg,100,150,50,60
path/to/image1.jpg,200,100,45,55
path/to/image2.jpg,50,80,70,80
```

Where `x1, y1` is the top-left corner and `w, h` are width/height in **pixels**. An image with empty `x1,y1,w,h` cells is treated as a negative-only (no face) image.

### Running Demos

```bash
# Webcam demo
python utils/webcam_demo.py --weights runs/checkpoints/BlazeFace_best.pth

# Image demo with ground truth comparison
python utils/image_demo.py --weights runs/checkpoints/BlazeFace_best.pth --csv data/splits/val.csv
```

---

## Project Structure

```text
trainable_blazeface/
├── blazebase.py          # Base classes, weight conversion, anchor generation
├── blazeface.py          # BlazeFace model (BlazeBlock_WT architecture)
├── blazedetector.py      # Inference utilities (NMS, box decoding)
├── dataloader.py         # CSV dataset, anchor encoding, augmentation
├── loss_functions.py     # Focal loss, Huber loss, hard negative mining
├── train_blazeface.py    # Training script with full pipeline
│
├── model_weights/
│   ├── blazeface.pth     # MediaPipe pretrained weights
│   └── anchors.npy       # Precomputed anchor coordinates
│
├── data/
│   ├── splits/           # Train/val CSV splits
│   └── raw/blazeface/    # Images organized by category
│
├── runs/
│   ├── checkpoints/      # Saved model weights
│   └── logs/             # TensorBoard logs
│
└── utils/
    ├── anchor_utils.py   # Vectorized anchor operations
    ├── augmentation.py   # Data augmentation functions
    ├── box_utils.py      # Box format conversions
    ├── config.py         # Default paths and settings
    ├── drawing.py        # Visualization utilities
    ├── iou.py            # IoU computation (batch/single)
    ├── metrics.py        # Precision, recall, mAP
    ├── model_utils.py    # Model loading helpers
    ├── webcam_demo.py    # Real-time webcam detection
    ├── image_demo.py     # Image detection with GT comparison
    └── debug_training.py # Training visualization tools
```

---

## Design Decisions

### Why Keep Keypoint Heads?

MediaPipe's BlazeFace outputs both **bounding boxes** and **6 facial keypoints**. We keep the keypoint heads in the architecture (frozen during training) because:

1. **Weight compatibility**: Dropping layers would break MediaPipe weight loading
2. **Future flexibility**: Keypoints can be trained later with appropriate data
3. **Minimal overhead**: Frozen heads add negligible computation

### Why Not Use the Paper's Architecture?

The original BlazeFace paper describes "double" BlazeBlocks for the back-facing model. MediaPipe's actual implementation is simpler:

- **Single BlazeBlocks** throughout
- **Two feature pyramid levels** (16×16 and 8×8)
- **Fixed anchor sizes** (predictions scaled by input size only)

We match MediaPipe's implementation exactly to ensure weight compatibility.

### Box Format: MediaPipe Convention

We use `[ymin, xmin, ymax, xmax]` (normalized 0-1) throughout, matching MediaPipe's internal format. This avoids conversion errors and makes debugging easier.

---

## References

### Papers

1. **BlazeFace**: Bazarevsky, V., et al. "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs." *CVPR Workshop on Computer Vision for AR/VR*, 2019. [[arXiv]](https://arxiv.org/abs/1907.05047)

2. **Focal Loss**: Lin, T., et al. "Focal Loss for Dense Object Detection." *ICCV*, 2017. [[arXiv]](https://arxiv.org/abs/1708.02002)

3. **MobileNets**: Howard, A., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." 2017. [[arXiv]](https://arxiv.org/abs/1704.04861)

### Code

This implementation builds on prior work:

- **[hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)**: PyTorch port of BlazeFace inference. This implementation provides weight loading and inference but does not support training due to folded batch normalization.
- **[vincent1bt/blazeface-tensorflow](https://github.com/vincent1bt/blazeface-tensorflow)**: TensorFlow training implementation. Note that this trains from scratch and does not load MediaPipe pretrained weights.
- **[zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch/)**: Additional MediaPipe model conversions.
- **[google/mediapipe](https://github.com/google/mediapipe)**: Original BlazeFace implementation.

### Dataset

- **[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)**: Yang, S., et al. "WIDER FACE: A Face Detection Benchmark." *CVPR*, 2016.
- **[Kaggle Face Detection Dataset](https://www.kaggle.com/datasets/ngoduy/dataset-for-face-detection)**: Additional face detection dataset with diverse annotations.
- **[LFPW Dataset](https://www.kaggle.com/datasets/amitmondal98/lfpw-labelled-face-parts-in-the-wild/data)**: Labeled Face Parts in the Wild dataset for facial landmark detection.

### Annotation Generation with RetinaFace

We use **[serengil/retinaface](https://github.com/serengil/retinaface)** to automatically generate face detection annotations for custom datasets. This allows you to create training data from any image collection without manual labeling.

The `data_prep.py` script scans an image directory, runs RetinaFace detection, and outputs CSV annotations compatible with our training pipeline:

```bash
# Generate annotations for all images in a directory
python data_prep.py --image-dir data/raw/blazeface/ --threshold 0.9

# This creates:
#   data/splits/retinaface_master.csv  (all detections)
#   data/splits/train.csv              (80% training split)
#   data/splits/val.csv                (20% validation split)
```

**Key options:**

| Flag | Description | Default |
| --- | --- | --- |
| `--image-dir` | Directory containing images to scan | `data/raw/blazeface/` |
| `--threshold` | Detection confidence threshold | `0.9` |
| `--val-fraction` | Fraction of images for validation | `0.2` |
| `--allow-upscaling` | Allow RetinaFace to upscale small images | `False` |

**Flags added in v2:** `--include-empty` writes images with no detection as negative-only rows; `--raw-boxes` keeps RetinaFace boxes unchanged (by default the top 10% is trimmed and the box widened to square).

**Output CSV format:**

```csv
image_path,x1,y1,w,h
0--Parade/image.jpg,120,45,80,95
```

This workflow enables knowledge distillation—training the lightweight BlazeFace using pseudo-labels from the more accurate (but slower) RetinaFace detector.

---

## Known Limitations

- **Pseudo-labels.** Training and validation labels come from RetinaFace (optionally edited, see `--raw-boxes`). Metrics measure agreement with that teacher, not real accuracy; faces RetinaFace misses count as errors for the student when it finds them.
- **No negative images by default.** `data_prep.py` only writes images with ≥ 1 detection unless `--include-empty` is set, so the false-positive rate on face-free images is neither trained on nor measured by default.
- **Train/val split.** A random per-image split of one pool; it can contain duplicate or near-duplicate images across splits (use the official WIDER FACE splits or perceptual-hash deduplication for real claims).
- **Metric semantics.** "AP@0.5" in the trainer is a per-image AP averaged over images, and "accuracy" is per anchor. Neither is comparable to published detection benchmarks; use the detection-level precision/recall tool for that.
- **Checkpoint selection** is by hard-negative-mined validation loss, which is only a proxy for detection quality. The trainer now validates on the full val set by default (`--val-max-images N` restricts it) and reports final metrics on the *best* checkpoint.
- **The shipped checkpoint predates v2** and underperforms the frozen MediaPipe weights (see above). Retrain with v2 before drawing conclusions about the loss or data choices.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---
