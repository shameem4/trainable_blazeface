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

> **Status: fine-tuning on this dataset does not beat the frozen MediaPipe weights.** Re-measured 2026-10-15. The root cause is a dataset/architecture mismatch, documented below — it is not a bug in the training code, and the v2 fixes did not change it.

### The port is faithful — verified against Google's own model

Before trusting any baseline, the implementation was checked against the official
`blaze_face_short_range.tflite` running under Google's `mediapipe.tasks` FaceDetector,
on identical images and ground truth (1,000-image random sample of val):

| Detector | Recall | Precision |
| --- | ---: | ---: |
| Google reference (official `.tflite`) | 0.4954 | 0.9398 |
| This repo's port, stock weights | 0.4936 | 0.9420 |

Agreement is within 4 true positives out of 1,125, with per-size-bin agreement.
Generated anchors are bit-identical to the reference `model_weights/anchors.npy`.
**So the stock baseline is sound, and the degradation below is real rather than a porting artifact.**

### Why fine-tuning hurts: half the labels are unresolvable

Ground-truth face size, expressed at the model's 128×128 letterboxed input:

| Face size @128px input | Train | Val | Stock recall |
| --- | ---: | ---: | ---: |
| 0–4 px | 21.4% | 17.2% | 0.000 |
| 4–8 px | 21.8% | 22.2% | 0.000 |
| 8–16 px | 13.5% | 13.9% | 0.081 |
| 16–32 px | 23.7% | 25.6% | 0.857 |
| 32+ px | 19.7% | 21.1% | 0.973 |
| **median** | **11.0 px** | **13.4 px** | |

The finest anchor stride is 8 px, so roughly 39% of labelled faces are below what
the architecture can resolve at all. Note that **46.7% of val boxes clear 16 px and
stock MediaPipe's measured overall recall is 0.4954** — those two numbers agreeing to
within a point is the evidence that the stock model is already at its architectural
ceiling on this data.

The labels are RetinaFace pseudo-labels over WIDER-style crowd images, while Google
documents this model as *"optimized for front-facing phone camera images at short
range"*. For about half the data, the label distribution and the architecture's
operating range do not overlap. Training with `detection_weight=200` forces the model
to chase positives it cannot represent, and the damage lands on the faces it previously
detected well.

### Detection-level results (IoU ≥ 0.5, unfiltered val, 2,000-image random sample)

`min16` is a retrain with ground-truth boxes below 16 px dropped, and images that would
lose more than half their faces removed entirely (so real faces never become background
labels). All three models are scored on the **unfiltered** val split — a filtered val
would flatter the filtered model.

| Model | Thr | Precision | Recall | 8–16 px | 16–32 px | 32+ px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MediaPipe (stock) | 0.5 | 0.939 | 0.464 | 0.081 | 0.857 | **0.973** |
| MediaPipe (stock) | 0.75 | 0.990 | 0.426 | 0.018 | 0.768 | 0.955 |
| Fine-tuned, unfiltered labels | 0.5 | **0.199** | 0.487 | 0.264 | 0.841 | 0.920 |
| Fine-tuned, unfiltered labels | 0.75 | 0.937 | 0.303 | 0.077 | 0.604 | 0.565 |
| Fine-tuned, `min16` labels | 0.5 | 0.912 | 0.448 | 0.028 | **0.869** | 0.923 |
| Fine-tuned, `min16` labels | 0.75 | 0.985 | 0.294 | 0.006 | 0.610 | 0.566 |

Reading this table:

- **The size filter fixes the precision collapse.** Training on unfiltered labels yields
  0.199 precision at threshold 0.5 — about 23 false positives per image. The filtered
  retrain recovers this to 0.912, a 4.6× improvement, confirming the diagnosis.
- **It still does not beat stock.** At comparable precision the filtered model recalls
  0.448 against stock's 0.464, and on the largest faces it reaches 0.923 against 0.973.
  Fine-tuning here buys nothing over the weights it started from.
- **The one genuine win is the 16–32 px band**, where the filtered model reaches 0.869
  against stock's 0.857 — the size range closest to the filter boundary.
- **Scores are not calibrated like stock's.** Both fine-tuned models fall off much faster
  with threshold, so comparing at a fixed threshold flatters stock; compare at matched
  precision instead.

### Things that did not work

Four further attempts, all scored on the unfiltered val split. None beats stock.

| @ thr 0.5 | precision | recall | 16-32 px | 32+ px |
| --- | ---: | ---: | ---: | ---: |
| **stock MediaPipe** | 0.939 | **0.464** | 0.857 | **0.973** |
| fine-tune, `min16` labels | 0.912 | 0.448 | **0.869** | 0.923 |
| frozen backbone, `min16` | 0.680 | 0.372 | 0.580 | 0.894 |
| single-level assignment, `min16` | **0.953** | 0.304 | 0.705 | 0.446 |
| + oversample large faces 4x | 0.927 | 0.250 | 0.558 | 0.415 |

**Freezing the backbone does not help.** It is worse than full fine-tuning in both
label conditions. The damage is not catastrophic forgetting of features -- the
backbone has no BatchNorm, so a frozen backbone is bit-identical to stock throughout
-- it is in the *heads*. Freezing protects the feature extractor, which was never
the problem, and concentrates all adaptation pressure into ~3,800 parameters.

**The anchor level assignment is a real defect, and fixing it makes things worse.**
`encode_boxes_to_anchors` assigns every box to *both* grids. Since the 8x8 grid has
6 anchors per cell against the 16x16 grid's 2, 73% of positive targets land in the
coarse grid -- where stock fires on 0.41% of them, against 9.09% in the fine grid.
Correcting it (`single_level=True`) drops stock's loss from 424 to 96 and lifts
positive accuracy from 0.148 to 0.410 *with nothing trained*, which is strong
evidence the objective was misaligned. But the trained A/B goes the other way:
+0.041 precision for -0.144 recall and -0.477 on 32+ px faces. The dual assignment
is wrong and load-bearing -- it keeps the coarse grid trained on every box. It
therefore **defaults to `False`**; the option and the measurements are documented in
the function's docstring.

**Oversampling does not rescue it.** Repeating large-face images 4x takes the coarse
grid from 6,919 boxes (4.09:1 fine:coarse) to 27,676 (1.10:1) and makes every metric
worse, including the 32+ px recall it targets. Large-face recall is not limited by
sample count.

> **The training objective is misaligned with detection quality on this data.**
> Stock MediaPipe -- the best detector here -- scores **424** on the training loss
> while the fine-tuned model scores **6.7**. 96.5% of stock's loss is the
> positive-classification term, because positive anchors are assigned to every GT
> box including the ~39% that are too small to resolve. Do not use val loss, IoU or
> AP to judge a checkpoint here; the single-level run posted the best training
> metrics of any run (AP 0.970, IoU 0.826) alongside the worst recall.

### Training-time metrics (full val split)

Training metrics improve dramatically under the filter while detection quality does not,
which is exactly why they should not be used to judge this pipeline.

| Metric | Pre-v2 ckpt | v2, unfiltered | v2, `min16` | Definition |
| --- | ---: | ---: | ---: | --- |
| Val loss | 6.70 | 6.63 | 5.33 | Hard-negative-mined; not comparable across loss settings |
| Positive accuracy | 73.5% | 73.9% | 74.8% | Positive *anchors* (8 per box) scoring > 0.45 |
| Background accuracy | 96.1% | 96.4% | 99.8% | Over all negative anchors, dominated by easy ones |
| Mean IoU | 0.578 | 0.582 | 0.784 | Decoded box vs GT on positive anchors (pre-NMS) |
| AP@0.5 | 0.758 | 0.760 | 0.957 | **Per-image** 11-point AP; not COCO/WIDER mAP |

> The `min16` column is measured on the filtered val split and is **not** comparable to the
> other two. Its AP of 0.957 sits alongside a detection-level result that loses to stock.

### Reproducing

```bash
# Build the size-filtered splits
python utils/filter_by_face_size.py --csv data/splits/train.csv --out data/splits/train_min16.csv
python utils/filter_by_face_size.py --csv data/splits/val.csv   --out data/splits/val_min16.csv

# Size-filtered retrain (~18 min on one GPU)
python train_blazeface.py --train-data data/splits/train_min16.csv \
    --val-data data/splits/val_min16.csv --init-weights mediapipe \
    --epochs 20 --checkpoint-dir runs/checkpoints_v2_min16

# Detection-level comparison on the unfiltered split, re-running the detector
# at each operating point (required: weighted NMS is threshold-coupled)
python utils/compare_detectors.py --sample 2000 --thresholds 0.5 0.75 0.9 \
    --weights mediapipe=model_weights/blazeface.pth \
              min16=runs/checkpoints_v2_min16/BlazeFace_best.pth

# Per-size-bin recall (add --by-size) is what separates resolvable faces from the rest
python utils/compare_detectors.py --sample 2000 --thresholds 0.5 --by-size \
    --weights mediapipe=model_weights/blazeface.pth

# Confirm the port still matches Google's official detector (needs `pip install mediapipe`)
curl -L -o /tmp/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
python utils/verify_against_reference.py --reference-model /tmp/blaze_face_short_range.tflite
```

Two measurement traps worth knowing, both of which produced wrong numbers during this work:

- **`val.csv` is ordered by category.** A head slice is all crowd scenes; sample randomly.
- **Weighted NMS is threshold-coupled.** `_weighted_non_max_suppression` blends every
  detection above `min_score_thresh`, so a PR curve cannot be swept from one low-threshold
  pass. Re-run the detector at each operating point.

### Debug Frames

Gray boxes are the ground-truth CSV, green are detections; produced by `utils/debug_training.py`.

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
- **Dataset/architecture mismatch is the dominant problem.** About 39% of labelled faces are under 8 px at the 128x128 input, below the finest anchor stride, while this model is designed for short-range selfie framing. Fine-tuning on the unfiltered labels is *worse than not fine-tuning at all*. The `min16` filter fixes the resulting precision collapse (0.199 -> 0.912) but still does not beat the stock weights. Treat this repo as a working training harness, not as a recipe that improves on MediaPipe for this data.
- **No checkpoint here beats frozen MediaPipe.** This holds across five configurations: the pre-v2 checkpoint, the v2 retrain, the `min16` retrain, a frozen backbone, and single-level anchor assignment with and without oversampling. If you need a better face detector for crowd imagery, a model with a finer stride and a larger input is the right answer, not more training of this one.
- **The v2 and `min16` checkpoints are not shipped.** Both lost to the stock weights, so keeping them would only invite someone to use them. Regenerate either in ~18 min with the commands under [Reproducing](#reproducing); `runs/logs_v2*` keeps the training curves behind the numbers above.
- **Filter side effects are unmeasured.** `min16` cut boxes by 60% but images by only 26.5% (2.70 -> 1.48 boxes/image), so it thinned crowded frames rather than removing them — the benign case. Whether pose or lighting variety narrowed along with size was never checked, so the `min16` numbers carry that caveat.
- **The 16-32 px win is one bin on one seed.** `min16` scores 0.869 there against stock's 0.857. It sits adjacent to the chosen filter boundary, has no confidence interval, and +1.2 points is the size of effect that routinely fails to replicate. Do not build on it without a second seed. The precision recovery (0.199 -> 0.912) is the robust result from that run.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---
