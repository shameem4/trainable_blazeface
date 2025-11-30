# Ear Teacher - Self-Supervised Learning

Self-supervised learning framework for extracting intricate ear features using a pretrained ConvNeXt backbone. The learned representations will be distilled to detector and landmarker models.

## Overview

The Ear Teacher model uses:
- **Backbone**: Pretrained ConvNeXt-Tiny (ImageNet-22k)
- **Learning**: SimSiam-style self-supervised learning
- **Input**: Cropped ear images with 10% bbox buffer
- **Framework**: PyTorch Lightning for clean, modular training

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
    --precision bf16-mixed \
    --devices 1 \
    --experiment_name my_experiment
```

### Key Arguments

**Data**:
- `--train_metadata`: Path to training metadata (default: `data/preprocessed/train_teacher.npy`)
- `--val_metadata`: Path to validation metadata (default: `data/preprocessed/val_teacher.npy`)
- `--batch_size`: Batch size (default: 32)
- `--image_size`: Input image size (default: 224)
- `--bbox_buffer`: Bbox buffer percentage (default: 0.10 for 10%)
- `--augment`: Enable augmentations (disabled by default)

**Model**:
- `--pretrained_path`: Path to pretrained weights (default: `models/convnext_tiny_22k_224.pth`)
- `--embedding_dim`: ConvNeXt output dimension (default: 768)
- `--projection_dim`: Projection head dimension (default: 256)
- `--freeze_backbone`: Freeze backbone initially

**Training**:
- `--learning_rate`: Peak learning rate (default: 1e-4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--warmup_epochs`: Warmup epochs (default: 10)
- `--max_epochs`: Maximum epochs (default: 100)
- `--precision`: Training precision (default: 16-mixed)

**Output**:
- `--output_dir`: Output directory (default: `outputs/ear_teacher`)
- `--experiment_name`: Experiment name (default: `default`)

## Architecture

### Model Components

1. **Backbone**: ConvNeXt-Tiny
   - Pretrained on ImageNet-22k
   - Outputs 768-dimensional embeddings
   - Can be frozen initially for transfer learning

2. **Projection Head**:
   - Maps embeddings to 256-dimensional space
   - Used for contrastive learning

3. **Prediction Head**:
   - SimSiam-style predictor
   - Prevents collapse in self-supervised learning

### Self-Supervised Learning

Currently uses SimSiam approach:
- Forward pass produces embeddings, projections, and predictions
- Loss: Negative cosine similarity between predictions and projections
- Symmetric loss: D(p1, z2) + D(p2, z1)

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
- **Checkpoints**: `outputs/ear_teacher/{experiment}/checkpoints/`
  - Best models (top-3 by validation loss)
  - Last checkpoint
  - Periodic checkpoints (every 5 epochs)

- **Logs**: `outputs/ear_teacher/{experiment}/`
  - TensorBoard logs
  - Training metrics

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/ear_teacher
```

Metrics logged:
- `train/loss`: Training loss
- `val/loss`: Validation loss
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
