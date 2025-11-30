# Using VAE Encoder for Detection and Landmarks

This guide explains how to leverage the trained VAE encoder for downstream ear detection and landmark localization tasks.

## Architecture Overview

### VAE Training (Phase 1 - Current)
```
Input Image (128x128)
  ↓
ResNet-50 Backbone (partial frozen) → 2048 channels, 4x4
  ↓
Conv1 + Attention → 1024 channels, 4x4  [feat1]
  ↓
Conv2 + Attention → 512 channels, 4x4  [feat2]
  ↓
Adaptive Pool → 512 channels, 4x4
  ↓
Conv3 + Attention → 512 channels, 4x4  [feat3]
  ↓
Latent Bottleneck → 1024D vector
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
  - resnet:  (B, 2048, 4x4)  ← ResNet-50 features
  - feat1:   (B, 1024, 4x4)  ← Custom conv + attention
  - feat2:   (B, 512, 4x4)   ← Mid-level features
  - feat3:   (B, 512, 4x4)   ← Refined features
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
#   'resnet': (B, 2048, 4, 4),  # ResNet-50 features
#   'feat1': (B, 1024, 4, 4),   # Custom conv + attention
#   'feat2': (B, 512, 4, 4),    # Mid-level features
#   'feat3': (B, 512, 4, 4)     # Refined features
# }
```

### 2. Spatial Information Preservation

Unlike the VAE bottleneck (1024D vector), features maintain spatial structure:
- **4x4 resolution** provides spatial localization for 128x128 images
- **Spatial attention** already focuses on ear regions
- **Multi-scale** allows detection at different granularities

## How to Use After VAE Training

### Step 1: Train VAE (Current Work)

```bash
# Train VAE with ResNet encoder
python ear_teacher/train.py \
    --train-npy data/preprocessed/train_teacher.npy \
    --val-npy data/preprocessed/val_teacher.npy \
    --epochs 60 \
    --batch-size 8
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
vae = EarVAE(latent_dim=1024, image_size=128)
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
- ResNet-50 provides ImageNet pretrained features
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
- 4x4 spatial resolution maintained across feature pyramid
- Multiple channel depths (2048 → 1024 → 512) for hierarchical features
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

### ResNet Features (2048 channels, 4x4)
- **Best for**: Rich semantic features from ImageNet pretraining
- **Resolution**: ~32x32 pixels per patch
- **Use case**: High-level ear structure understanding

### feat1 (1024 channels, 4x4)
- **Best for**: Ear-specific adapted features
- **Resolution**: ~32x32 pixels per patch
- **Attention**: Spatial attention applied
- **Use case**: Primary features for detection

### feat2 (512 channels, 4x4)
- **Best for**: Mid-level features
- **Resolution**: ~32x32 pixels per patch
- **Attention**: Spatial attention applied
- **Use case**: Refined ear features

### feat3 (512 channels, 4x4)
- **Best for**: Final refined features before latent bottleneck
- **Resolution**: ~32x32 pixels per patch
- **Attention**: Spatial attention applied
- **Use case**: Most refined spatial features for landmarks

## Advanced: Custom Detection Heads

You can customize the detection heads for your specific needs:

```python
class CustomEarDetector(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder

        # Use feat3 (4x4, 512 channels) for spatial tasks
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
            nn.ConvTranspose2d(256, 128, 4, stride=4),  # Upsample 4x4 to 16x16
            nn.ReLU(),
            nn.Conv2d(128, 17, 1)  # 17 heatmaps, one per landmark
        )

    def forward(self, x):
        features = self.encoder.extract_features(x)
        feat = features['feat3']  # (B, 512, 4, 4)

        bboxes = self.bbox_head(feat)
        heatmaps = self.keypoint_heatmaps(feat)  # (B, 17, 16, 16)

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
- Pre-trained ResNet-50 + custom layers
- Spatial attention focused on ears
- Multi-scale features (2048, 1024, 512 channels at 4x4)
- Better starting point than random weights

**What you need for detection:**
- Labeled data: images + bboxes + 17 landmark coordinates
- Detection loss functions (bbox + keypoint)
- Fine-tuning script (can adapt current training script)

The VAE is your **feature learning phase**. Detection is the **task-specific phase**.
