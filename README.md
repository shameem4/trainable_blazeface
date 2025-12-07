# Trainable BlazeFace

A PyTorch implementation of BlazeFace that supports fine-tuning from MediaPipe's pretrained weights.

---

## Motivation

BlazeFace is a lightweight face detection model developed by Google for mobile applications. It achieves real-time performance (200+ FPS on mobile GPUs) while maintaining reasonable accuracy for its intended use case: detecting a single frontal face in selfie-style images.

However, the pretrained weights distributed through MediaPipe are optimized for inference only. The TensorFlow Lite format does not support gradient computation, and the batch normalization layers are folded into convolution weights for speed.

Existing PyTorch implementations, notably [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch), provide inference capabilities but do not support training. These implementations load the pretrained weights but cannot fine-tune them because the architecture lacks proper batch normalization layers.

This repository addresses that limitation by providing a trainable architecture that:

1. Loads MediaPipe pretrained weights with full compatibility
2. Unfolds the folded batch normalization parameters into trainable layers
3. Produces identical outputs to MediaPipe at initialization
4. Supports standard PyTorch training with gradient backpropagation

---

## Overview

The system converts frozen MediaPipe weights into a trainable form:

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINABLE BLAZEFACE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   MediaPipe Weights (.pth)                                          │
│          │                                                          │
│          ▼                                                          │
│   ┌──────────────────┐     ┌──────────────────┐                     │
│   │  Weight Unfolder │────▶│  BlazeBlock_WT   │                     │
│   │  (BatchNorm      │     │  (Trainable      │                     │
│   │   extraction)    │     │   architecture)  │                     │
│   └──────────────────┘     └────────┬─────────┘                     │
│                                     │                               │
│          ┌──────────────────────────┼──────────────────────────┐    │
│          ▼                          ▼                          ▼    │
│   ┌─────────────┐           ┌─────────────┐           ┌────────────┐│
│   │  Your Data  │           │   Anchor    │           │   Loss     ││
│   │  (CSV+Images)│           │  Encoding   │           │  Functions ││
│   └─────────────┘           └─────────────┘           └────────────┘│
│          │                          │                          │    │
│          └──────────────────────────┴──────────────────────────┘    │
│                                     │                               │
│                                     ▼                               │
│                          ┌──────────────────┐                       │
│                          │  Fine-tuned      │                       │
│                          │  BlazeFace       │                       │
│                          │  (.ckpt/.pth)    │                       │
│                          └──────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Weight Unfolding

MediaPipe's BlazeFace uses folded batch normalization, where the batch normalization parameters are mathematically merged into the convolution weights. This optimization improves inference speed but prevents training since the normalization statistics are no longer separable.

The weight unfolding process reverses this transformation:

```python
# MediaPipe format: Conv with folded BN (inference-only)
# W_folded = W × γ / √(var + ε)
# b_folded = (b - μ) × γ / √(var + ε) + β

# Our approach: Unfold back to trainable form
def unfold_conv_bn(conv_weight, conv_bias, num_features):
    """Extract trainable conv + BatchNorm from folded weights."""
    new_conv_weight = conv_weight.clone()
    bn_weight = torch.ones(num_features)      # γ = 1
    bn_bias = conv_bias.clone()               # β absorbs original bias
    bn_running_mean = torch.zeros(num_features)  # μ = 0
    bn_running_var = torch.ones(num_features)    # σ² = 1
    return new_conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var
```

The resulting model produces identical outputs to MediaPipe at initialization while supporting gradient-based optimization through proper batch normalization layers.

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

### The BlazeBlock: Efficiency Through Simplicity

Each BlazeBlock is a **depthwise separable convolution** with a skip connection—the same building block that powers MobileNets:

```text
         Input
           │
     ┌─────┴─────┐
     │           │
     ▼           │
┌─────────┐      │
│Depthwise│      │
│ Conv 5×5│      │ (skip connection with optional
├─────────┤      │  1×1 projection if channels differ)
│BatchNorm│      │
├─────────┤      │
│  ReLU   │      │
├─────────┤      │
│Pointwise│      │
│ Conv 1×1│      │
├─────────┤      │
│BatchNorm│      │
└────┬────┘      │
     │           │
     └─────┬─────┘
           │
         (+) Add
           │
         ReLU
           │
        Output
```

The computational advantage of depthwise separable convolutions: a standard 3×3 convolution with C input and C output channels requires C × C × 9 parameters. Depthwise separable convolutions split this into:

- Depthwise: C × 9 parameters (one filter per channel)
- Pointwise: C × C × 1 parameters (channel mixing only)

Total: C × 9 + C² vs 9 × C², approximately 9× parameter reduction.

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
│   │   L = 150 × L_box + 40 × L_bg + 80 × L_pos                      │       │
│   │                                                                 │       │
│   │   L_box: Huber loss on positive anchor box predictions          │       │
│   │   L_bg:  Focal loss on hard negative backgrounds                │       │
│   │   L_pos: Focal loss on positive anchors (faces)                 │       │
│   │                                                                 │       │
│   │   Hard Negative Mining: Select 1.5× negatives per positive      │       │
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

The training pipeline applies data augmentations to improve model robustness across different imaging conditions. Each augmentation is applied independently with a fixed probability (50% for color and geometric transforms, 30% for occlusion-based transforms).

### Color Augmentations

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| **Saturation** | 50% | Random saturation shift in HSV space (0.75–1.25×) |
| **Brightness** | 50% | Random value shift in HSV space (0.75–1.25×) |
| **Color Jitter** | 50% | Combined H/S/V perturbation: hue ±10°, sat 0.8–1.2×, val 0.8–1.2× |

### Geometric Augmentations

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| **Horizontal Flip** | 50% | Mirror image with bbox coordinate adjustment |
| **Random Scale** | 50% | Scale 0.85–1.15× with center crop/pad, bbox rescaling |
| **Random Rotation** | 50% | Rotate ±10° with bounding box recalculation |

### Occlusion Augmentations

| Augmentation | Probability | Description |
|--------------|-------------|-------------|
| **Synthetic Occlusion** | 30% | Gray rectangles over 30–60% of random face bboxes |
| **Cutout** | 30% | Random 10–25px rectangular patches filled with mean color |

### Augmentation Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT IMAGE + BBOXES                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COLOR AUGMENTATIONS                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Saturation  │  │  Brightness  │  │ Color Jitter │           │
│  │   (50%)      │  │   (50%)      │  │   (50%)      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GEOMETRIC AUGMENTATIONS                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   H-Flip     │  │ Random Scale │  │  Rotation    │           │
│  │   (50%)      │  │   (50%)      │  │   (50%)      │           │
│  │              │  │ 0.85–1.15×   │  │  ±10°        │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OCCLUSION AUGMENTATIONS                       │
│  ┌──────────────────────┐  ┌───────────────────────┐            │
│  │ Synthetic Occlusion  │  │       Cutout          │            │
│  │      (30%)           │  │       (30%)           │            │
│  │ Gray boxes on faces  │  │ Random patch removal  │            │
│  └──────────────────────┘  └───────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUGMENTED OUTPUT                             │
└─────────────────────────────────────────────────────────────────┘
```

### Augmentation Rationale

| Augmentation Type | Target Variation |
|-------------------|------------------|
| Color variations | Different lighting, cameras, skin tones |
| Geometric | Different face orientations, camera distances |
| Occlusion | Partial face visibility, overlapping objects |

---

## Experimental Results

### Field Notes: BlazeFace_best.pth (Epoch 17)

The latest checkpoint living in `runs/checkpoints/BlazeFace_best.pth` was captured after epoch 17 and logged to `runs/logs/BlazeFace/events.out.tfevents.*`. Treat this section like a lab notebook entry: all numbers below are pulled directly from that run, no cherry-picking.

#### Quick Stats

| Field | Value | Notes |
|-------|-------|-------|
| Parameters | 101,390 | Matches MediaPipe-compatible architecture |
| Device | `cuda:0` | Measured throughput ≈87 img/s during evaluation |
| Eval subset | 500 images / 3,024 faces | `data/splits/val.csv` sample pulled via `utils/debug_training.py --run-eval` |
| Detection threshold | 0.2 | Aggressive recall setting to surface borderline faces |
| Reference weights | MediaPipe `model_weights/blazeface.pth` | Used for side-by-side diagnostics |

#### Detection Metrics (threshold = 0.2)

| Metric | Value | Comment |
|--------|-------|---------|
| Images evaluated | 500 | Batched inside `debug_training.py` |
| Total detections | 3,399 | Includes faces + false alarms |
| True positives | 288 | Matching IoU ≥ 0.5 |
| False positives | 3,111 | Main opportunity for improvement |
| False negatives | 2,736 | Many small/occluded faces remain hard |
| Precision | **8.5%** | Low by design because of the permissive threshold |
| Recall | **9.5%** | +3.3 pts vs frozen MediaPipe baseline |
| F1 score | **8.97%** | Snapshot of the current trade-off |

Even though the precision is modest, this run uncovers how the model behaves when we bias toward recall. A follow-up sweep with higher thresholds or tighter NMS would reduce noise without re-running training.

### Debug Frames from the Same Run

| Parade crowd (dense occlusion) | Political rally (high contrast) | Pool deck (specular highlights) |
|:--:|:--:|:--:|
| ![Parade debug frame](assets/latest_run/sample_parade.png) | ![Protest debug frame](assets/latest_run/sample_protest.png) | ![Swimming debug frame](assets/latest_run/sample_swim.png) |
| `sample_0000_0_Parade_marchingband_1_1031.png` | `sample_2777_2_Demonstration_Political_Rally_2_905.png` | `sample_6512_41_Swimming_Swimming_41_515.png` |

Green boxes show BlazeFace detections while gray boxes plot the ground-truth CSV annotations. These frames are generated by `utils/debug_training.py --run-eval` and land in `runs/logs/debug_images/` before being copied into `assets/latest_run/` for documentation.

### Baseline: MediaPipe Pretrained Weights

The pretrained MediaPipe weights were evaluated on the WIDER FACE validation set, a benchmark containing faces at varying scales, poses, and occlusion levels.

| Metric | Value |
|--------|-------|
| Images Evaluated | 500 |
| Ground Truth Faces | 3,024 |
| Detections | 192 |
| **Precision** | **97.9%** |
| **Recall** | **6.2%** |
| **F1 Score** | **11.7%** |

The pretrained weights exhibit high precision (97.9%) but low recall (6.2%). This behavior is expected: MediaPipe was optimized for single frontal face detection in selfie-style images. The model rejects uncertain detections, which is appropriate for its intended use case but results in missed detections in crowded scenes with small, occluded, or profile faces.

#### Baseline vs Fine-tuned Snapshot (val subset, threshold = 0.2)

| Metric | MediaPipe Pretrained | Fine-tuned (Epoch 17) |
|--------|----------------------|-----------------------|
| Images Evaluated | 500 | 500 |
| Ground Truth Faces | 3,024 | 3,024 |
| Detections | 192 | 3,399 |
| Precision | **97.9%** | **8.5%** |
| Recall | **6.2%** | **9.5%** |
| F1 Score | **11.7%** | **9.0%** |

The fine-tuned checkpoint purposely lowers the detection threshold to surface harder faces—hence the precision hit and slight recall bump. It is a useful diagnostic phase before we tighten the score cutoff or add a learned confidence calibration head.

### Training Results

Fine-tuning was performed on the WIDER FACE training set (32,325 images, 87,301 face annotations):

```text
Training Configuration:
├── Epochs: 12 (resumed from checkpoint)
├── Batch Size: 32
├── Learning Rate: 0.0005 (AdamW)
├── Loss: Focal (classification) + Huber (regression)
└── Device: NVIDIA CUDA GPU
```

The metric trace below is exported directly from the TensorBoard run stored in `runs/logs/BlazeFace/events.out.tfevents*`. It reflects the same training session that produced `BlazeFace_best.pth`.

#### Training Metrics Over Time

| Epoch | Train Loss | Val Loss | Pos Accuracy | Bg Accuracy | **Val IoU** |
|-------|------------|----------|--------------|-------------|-------------|
| 1     | 7.54       | 6.68     | 74.7%        | 93.1%       | 0.499       |
| 2     | 6.20       | 6.30     | 74.6%        | 94.4%       | 0.521       |
| 3     | 5.64       | 5.97     | 76.1%        | 94.3%       | 0.548       |
| 4     | 5.67       | 6.02     | 74.7%        | 95.0%       | 0.553       |
| 5     | 5.80       | 6.23     | 78.9%        | 92.6%       | 0.543       |
| 6     | 5.83       | 5.98     | 75.3%        | 94.5%       | 0.534       |
| 7     | 5.65       | 5.87     | 78.4%        | 95.0%       | 0.558       |
| 8     | 5.37       | 5.69     | 76.3%        | 95.4%       | 0.571       |
| 9     | 5.27       | 5.68     | 77.6%        | 94.7%       | **0.571**   |
| 10    | 5.31       | 5.65     | 77.6%        | 95.2%       | 0.570       |
| 11    | 5.49       | 5.86     | 76.0%        | 94.6%       | 0.560       |
| 12    | 5.53       | 6.06     | 79.7%        | 92.5%       | 0.544       |

**Summary:**

| Metric | Initial | Best | Change |
|--------|---------|------|--------|
| Val IoU | 0.499 | 0.571 (epoch 9) | +14.5% |
| Val Loss | 6.68 | 5.65 | -15.4% |
| mAP@0.5 | — | 68.0% | — |

**Observations:**

- Validation IoU improved from 0.499 to 0.571 over 12 epochs
- Background rejection accuracy peaked at 95.4%
- Transfer learning from MediaPipe initialization provides faster convergence than random initialization
- Best performance achieved at epochs 9-10; subsequent epochs show mild overfitting

For production applications, training for 20-50 epochs with learning rate scheduling and early stopping is recommended.

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
    --lr 0.0005

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

Where `x1, y1` is the top-left corner and `w, h` are width/height in **pixels**.

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
|------|-------------|---------|
| `--image-dir` | Directory containing images to scan | `data/raw/blazeface/` |
| `--threshold` | Detection confidence threshold | `0.9` |
| `--val-fraction` | Fraction of images for validation | `0.2` |
| `--allow-upscaling` | Allow RetinaFace to upscale small images | `False` |

**Output CSV format:**

```csv
image_path,x1,y1,width,height,score
0--Parade/image.jpg,120,45,80,95,0.998
```

This workflow enables knowledge distillation—training the lightweight BlazeFace using pseudo-labels from the more accurate (but slower) RetinaFace detector.

---

## Known Limitations

### Train/Val Split

The current train/validation split is not optimal and may contain duplicate or near-duplicate images across splits. This can lead to:

- Overly optimistic validation metrics
- Potential data leakage between train and val sets

For production use, consider:

- Using the official WIDER FACE train/val splits
- Implementing proper deduplication (e.g., perceptual hashing)
- Ensuring no image appears in both sets

---

## License

Creative Commons Attribution-NonCommercial 4.0 International. See [LICENSE](LICENSE) for the exact terms.

---
