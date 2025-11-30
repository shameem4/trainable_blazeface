# Ear Teacher - Self-Supervised Learning

Self-supervised learning framework for extracting intricate ear features using a pretrained ConvNeXt backbone. The learned representations will be distilled to detector and landmarker models.

## Overview

The Ear Teacher model uses:
- **Backbone**: Pretrained ConvNeXt-Tiny (ImageNet-22k)
- **Learning**: Reconstruction + metric learning (ArcFace/CosFace)
- **Input**: Cropped ear images with 10% bbox buffer
- **Framework**: PyTorch Lightning for clean, modular training
- **Monitoring**: Automatic reconstruction collages after each validation epoch

## Project Structure

```
ear_teacher/
├── dataset.py           # Dataset loading with bbox cropping
├── datamodule.py        # Lightning DataModule
├── model.py             # ConvNeXt backbone + projection heads
├── lightning_module.py  # Training logic and loss
├── train.py            # Main training script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Installation

Install dependencies:

```bash
pip install -r ear_teacher/requirements.txt
```

## Data Format

Training expects metadata files at:
- `data/preprocessed/train_teacher.npy`
- `data/preprocessed/val_teacher.npy`

Each .npy file contains a dictionary with:
```python
{
    'image_paths': array of image paths (relative to root),
    'bboxes': array of [x1, y1, x2, y2] bbox coordinates
}
```

## Training

### Basic Training

Run from the root directory:

```bash
python -m ear_teacher.train
```

### Custom Configuration

```bash
python -m ear_teacher.train \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --max_epochs 200 \
    --reconstruction_weight 1.0 \
    --metric_weight 0.1 \
    --metric_loss arcface \
    --num_pseudo_classes 512 \
    --arcface_margin 0.5 \
    --precision bf16-mixed \
    --devices 1 \
    --experiment_name my_experiment
```

### Key Arguments

**Data**:
- `--train_metadata`: Path to training metadata (default: `data/preprocessed/train_teacher.npy`)
- `--val_metadata`: Path to validation metadata (default: `data/preprocessed/val_teacher.npy`)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: Dataloader workers (default: 8)
- `--image_size`: Input image size (default: 224)
- `--bbox_buffer`: Bbox buffer percentage (default: 0.10 for 10%)
- `--augment`: Enable augmentations (disabled by default)

**Model**:
- `--pretrained_path`: Path to pretrained weights (default: `models/convnext_tiny_22k_224.pth`)
- `--embedding_dim`: ConvNeXt output dimension (default: 768)
- `--projection_dim`: Projection head dimension (default: 256)

**Training**:
- `--learning_rate`: Peak learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--warmup_epochs`: Warmup epochs (default: 10)
- `--max_epochs`: Maximum epochs (default: 100)
- `--reconstruction_weight`: Reconstruction loss weight (default: 1.0)
- `--metric_weight`: Metric learning loss weight (default: 0.1)
- `--metric_loss`: Type of metric loss - 'arcface', 'cosface', or 'none' (default: arcface)
- `--num_pseudo_classes`: Number of pseudo-classes for metric learning (default: 512)
- `--arcface_margin`: Angular margin for ArcFace/CosFace (default: 0.5)
- `--arcface_scale`: Feature scale for ArcFace/CosFace (default: 64.0)
- `--num_collage_samples`: Samples for validation collage (default: 10)
- `--precision`: Training precision (default: 16-mixed)

**Output**:
- `--output_dir`: Output directory (default: `ear_teacher/outputs/`)
- `--experiment_name`: Experiment name (default: `ear_teacher`)

## Architecture

### Model Components

1. **Backbone**: ConvNeXt-Tiny
   - Pretrained on ImageNet-22k
   - Outputs 768-dimensional embeddings
   - Fully trainable (no freezing)

2. **Projection Head**:
   - Maps embeddings to 256-dimensional space
   - Used for contrastive learning

3. **Prediction Head**:
   - SimSiam-style predictor
   - Prevents collapse in self-supervised learning

4. **Reconstruction Decoder**:
   - Upsamples embeddings back to image space (224x224)
   - Uses transposed convolutions with batch normalization
   - Learns to reconstruct original ear images

### Self-Supervised Learning

Uses combined reconstruction + metric learning:

1. **Reconstruction Loss** (MSE):
   - Learns to reconstruct input images from embeddings
   - Encourages embeddings to capture intricate ear details
   - Default weight: 1.0

2. **Metric Learning Loss** (ArcFace/CosFace):
   - **ArcFace** (default): Additive angular margin loss for discriminative embeddings
     - Adds angular penalty to make embeddings more separable
     - Margin: 0.5 (controls inter-class separation)
     - Scale: 64.0 (controls feature magnitude)
   - **CosFace**: Large margin cosine loss (simpler alternative to ArcFace)
   - Uses pseudo-labels (512 classes by default) for self-supervised training
   - Learns discriminative features that can distinguish between different ears
   - Default weight: 0.1

3. **Validation Monitoring**:
   - After each epoch, creates a collage of 10 random validation samples
   - Shows original images (top row) vs reconstructions (bottom row)
   - Saved to `{log_dir}/reconstruction_collages/epoch_XXXX.png`

**Why ArcFace/CosFace?**
- Originally designed for face recognition, excellent for ear recognition too
- Learns embeddings where similar ears cluster together
- Better than standard contrastive learning for identification tasks
- Will produce features ideal for distillation to detector/landmarker models

### Data Processing

1. Load image from metadata
2. Add 10% buffer to bbox coordinates
3. Crop to buffered bbox
4. Resize to 224x224
5. Normalize with ImageNet statistics
6. Apply augmentations (when enabled)

## Augmentation Strategy

**Current**: No augmentations (baseline)

**Future** (gradually enable in `dataset.py`):
- Horizontal flip
- Small rotations and shifts
- Brightness/contrast adjustments
- Gaussian noise

## Outputs

Training produces:
- **Checkpoints**: `ear_teacher/outputs/{experiment}/checkpoints/`
  - Best models (top-3 by validation loss)
  - Last checkpoint
  - Periodic checkpoints (every 5 epochs)

- **Logs**: `ear_teacher/outputs/{experiment}/`
  - TensorBoard logs
  - Training metrics

- **Reconstruction Collages**: `ear_teacher/outputs/{experiment}/*/reconstruction_collages/`
  - Generated after each validation epoch
  - 2-row grid: originals (top) vs reconstructions (bottom)
  - 10 samples per collage (configurable with `--num_collage_samples`)
  - Filename: `epoch_XXXX.png`

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir ear_teacher/outputs
```

Metrics logged:
- `train/loss`: Total training loss
- `train/recon_loss`: Reconstruction loss (training)
- `train/metric_loss`: Metric learning loss - ArcFace/CosFace (training)
- `val/loss`: Total validation loss
- `val/recon_loss`: Reconstruction loss (validation)
- `val/metric_loss`: Metric learning loss - ArcFace/CosFace (validation)
- `train/lr`: Learning rate

## Next Steps

1. **Baseline Training**: Train without augmentations to establish baseline
2. **Add Augmentations**: Gradually enable augmentations in `dataset.py`
3. **Feature Extraction**: Use trained backbone to extract ear features
4. **Distillation**: Distill knowledge to detector and landmarker models

## Usage Example

```python
from ear_teacher import (
    EarDataModule,
    EarTeacherLightningModule,
)
import pytorch_lightning as pl

# Setup data
datamodule = EarDataModule(
    batch_size=32,
    num_workers=4,
)

# Setup model
model = EarTeacherLightningModule(
    learning_rate=1e-4,
    max_epochs=100,
)

# Train
trainer = pl.Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, datamodule=datamodule)
```

## License

Internal research project.
