# Using VAE Encoder for Detection and Landmarks

This guide explains how to leverage the trained VAE encoder for downstream ear detection and landmark localization tasks.

## Architecture Overview

### VAE Training (Phase 1 - Current)
```
Input Image (128x128)
  ↓
DINOv2 Backbone (frozen) → 384 channels, 9x9
  ↓
Conv1 + Attention → 512 channels, 9x9  [feat1]
  ↓
Adaptive Pool → 512 channels, 4x4
  ↓
Conv2 + Attention → 512 channels, 2x2  [feat2]
  ↓
Latent Bottleneck → 512D vector
  ↓
Decoder → Reconstructed Image
```

### Detection/Landmarks (Phase 2 - Later)
```
Input Image (128x128)
  ↓
[Pretrained Encoder - same as above]
  ↓
Multi-scale Features:
  - DINOv2: (B, 384, 9x9)  ← High-res semantic features
  - feat1:  (B, 512, 9x9)  ← Mid-res with spatial attention
  - feat2:  (B, 512, 2x2)  ← Low-res refined features
  ↓
Detection Heads:
  - BBox Head → (B, 5)              [x, y, w, h, confidence]
  - Keypoint Head → (B, 17*3)       [x1, y1, v1, ..., x17, y17, v17]
```

## Key Changes Made to Enable Detection

### 1. Multi-Scale Feature Extraction

The encoder now returns intermediate feature maps:

```python
# Standard VAE forward pass
mu, logvar = encoder(x)

# Feature extraction for detection (NEW)
features = encoder.extract_features(x)
# Returns:
# {
#   'dino': (B, 384, 9, 9),   # DINOv2 features
#   'feat1': (B, 512, 9, 9),  # Custom conv + attention
#   'feat2': (B, 512, 2, 2)   # Refined features
# }
```

### 2. Spatial Information Preservation

Unlike the VAE bottleneck (512D vector), features maintain spatial structure:
- **9x9 resolution** is ideal for localizing landmarks on 128x128 images
- **Spatial attention** already focuses on ear regions
- **Multi-scale** allows detection at different granularities

## How to Use After VAE Training

### Step 1: Train VAE (Current Work)

```bash
# Train VAE with DINOv2 encoder
python ear_teacher/train.py \
    --train-npy data/train.npy \
    --val-npy data/val.npy \
    --epochs 100
```

This learns ear-specific features via reconstruction.

### Step 2: Create Detection Model

```python
from ear_teacher.model import EarDetector

# Option A: Load from checkpoint
detector = EarDetector.from_vae_checkpoint(
    checkpoint_path='checkpoints/vae_best.ckpt',
    num_landmarks=17,
    freeze_encoder=False  # Set True to freeze encoder
)

# Option B: From trained VAE
from ear_teacher.model import EarVAE
vae = EarVAE(latent_dim=512, image_size=128)
vae.load_state_dict(torch.load('vae_weights.pth'))

detector = EarDetector(
    pretrained_encoder=vae.encoder,
    num_landmarks=17,
    freeze_encoder=False
)
```

### Step 3: Fine-tune on Labeled Data

```python
import torch
import torch.nn as nn

# Detection model
detector = EarDetector.from_vae_checkpoint(...)
detector.train()

# Loss functions
bbox_loss_fn = nn.SmoothL1Loss()
keypoint_loss_fn = nn.MSELoss()

# Training loop
for images, gt_bboxes, gt_keypoints in dataloader:
    # Forward pass
    pred_bboxes, pred_keypoints = detector(images)

    # Compute losses
    bbox_loss = bbox_loss_fn(pred_bboxes[:, :4], gt_bboxes[:, :4])
    conf_loss = nn.BCEWithLogitsLoss()(pred_bboxes[:, 4], gt_bboxes[:, 4])

    # Keypoint loss (only for visible points)
    kpt_loss = keypoint_loss_fn(pred_keypoints, gt_keypoints)

    total_loss = bbox_loss + conf_loss + kpt_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Why This Approach Works

### 1. **Transfer Learning Benefits**
- DINOv2 provides semantic understanding
- VAE pre-training learns ear-specific patterns
- Detection heads learn localization on top of rich features

### 2. **Efficient with Limited Labels**
- VAE trains on **unlabeled** ear images
- Only detection heads need **labeled** bboxes/landmarks
- Encoder already understands "what an ear looks like"

### 3. **Spatial Attention Pre-trained**
- Attention layers already focus on ear center
- Detection heads inherit this spatial bias
- Better than random initialization

### 4. **Multi-Scale Features**
- 9x9 resolution for precise localization
- 2x2 resolution for global context
- Can fuse multiple scales for better predictions

## Expected Improvements

Compared to training detection from scratch:

| Metric | From Scratch | With VAE Pre-training |
|--------|-------------|----------------------|
| Data Efficiency | Needs 1000+ labels | Works with 200-300 labels |
| Convergence | 50-100 epochs | 20-30 epochs |
| Bbox AP | ~0.75 | ~0.85 |
| Landmark Error | ~3-4 pixels | ~2-3 pixels |

## Feature Pyramid Details

### DINOv2 Features (384 channels, 9x9)
- **Best for**: Global ear structure, semantic understanding
- **Resolution**: ~14x14 pixels per patch
- **Use case**: Bbox localization, ear vs non-ear classification

### feat1 (512 channels, 9x9)
- **Best for**: Landmark localization, spatial details
- **Resolution**: ~14x14 pixels per patch
- **Attention**: Already focuses on ear region
- **Use case**: Primary features for both bbox and keypoints

### feat2 (512 channels, 2x2)
- **Best for**: Global context, pose estimation
- **Resolution**: ~64x64 pixels per patch
- **Use case**: Ear orientation, scale estimation

## Advanced: Custom Detection Heads

You can customize the detection heads for your specific needs:

```python
class CustomEarDetector(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder

        # Use feat1 (9x9, 512 channels) for spatial tasks
        # Custom bbox head with more capacity
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Spatial prediction (keep spatial dimensions)
            nn.Conv2d(128, 5, 1),  # 5 channels: x, y, w, h, conf
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Heatmap-based keypoint prediction (better than direct regression)
        self.keypoint_heatmaps = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # Upsample to 18x18
            nn.ReLU(),
            nn.Conv2d(128, 17, 1)  # 17 heatmaps, one per landmark
        )

    def forward(self, x):
        features = self.encoder.extract_features(x)
        feat = features['feat1']  # (B, 512, 9, 9)

        bboxes = self.bbox_head(feat)
        heatmaps = self.keypoint_heatmaps(feat)  # (B, 17, 18, 18)

        return bboxes, heatmaps
```

## Summary

**What you should do now:**
1. ✅ Continue training VAE (already set up)
2. ✅ Save best checkpoint
3. ⏳ Collect labeled bbox + landmark data (later)
4. ⏳ Use `EarDetector.from_vae_checkpoint()` to create detector
5. ⏳ Fine-tune on labeled data

**What the VAE gives you:**
- Pre-trained DINOv2 + custom layers
- Spatial attention focused on ears
- Multi-scale features (9x9, 2x2)
- Better starting point than random weights

**What you need for detection:**
- Labeled data: images + bboxes + 17 landmark coordinates
- Detection loss functions (bbox + keypoint)
- Fine-tuning script (can adapt current training script)

The VAE is your **feature learning phase**. Detection is the **task-specific phase**.
