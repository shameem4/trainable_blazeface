# BlazeFace Training Guide

This guide explains how to train the BlazeFace ear detector using the CSV dataset.

## Dataset Preparation

### 1. CSV Format
The dataset uses CSV format with the following columns:
- `image_path`: Relative path to image
- `x1`, `y1`, `w`, `h`: Bounding box in pixel coordinates
- `width`, `height`: Original image dimensions

### 2. Train/Val Split
Split your CSV dataset into training and validation sets:

```bash
python csv_dataloader.py \
    --csv data/raw/blazeface/fixed_images.csv \
    --output data/splits \
    --val-split 0.2 \
    --seed 42
```

This creates:
- `data/splits/train.csv` (80% of data)
- `data/splits/val.csv` (20% of data)

## Training

### Basic Training
```bash
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --epochs 500 \
    --batch-size 32
```

### Training Parameters

#### Data Arguments
- `--csv-format`: Enable CSV format (required for CSV data)
- `--train-data`: Path to training CSV file
- `--val-data`: Path to validation CSV file (optional)
- `--data-root`: Root directory for image paths (required with CSV)

#### Model Arguments
- `--pretrained`: Load pretrained MediaPipe weights

#### Training Arguments
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 500)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 1e-4)

#### Loss Arguments
- `--use-focal-loss`: Use focal loss instead of BCE
- `--focal-alpha`: Focal loss alpha (default: 0.25)
- `--focal-gamma`: Focal loss gamma (default: 2.0)
- `--detection-weight`: Weight for box regression loss (default: 150.0)
- `--classification-weight`: Weight for classification loss (default: 35.0)
- `--hard-negative-ratio`: Ratio of negatives to positives (default: 3)

#### System Arguments
- `--device`: Device (cuda or cpu, default: cuda)
- `--num-workers`: Data loading workers (default: 4)
- `--checkpoint-dir`: Checkpoint directory (default: checkpoints)
- `--log-dir`: TensorBoard log directory (default: logs)

### Auto-Resume
The trainer automatically resumes from the best checkpoint if it exists:

```bash
# This will auto-resume if checkpoints/BlazeFace_best.pth exists
python train_blazeface.py --csv-format --train-data data/splits/train.csv ...
```

To start fresh and disable auto-resume:
```bash
python train_blazeface.py --csv-format --train-data data/splits/train.csv ... --no-auto-resume
```

To resume from a specific checkpoint:
```bash
python train_blazeface.py --csv-format --train-data data/splits/train.csv ... --resume checkpoints/BlazeFace_epoch100.pth
```

## Monitoring

### TensorBoard
View training progress in real-time:

```bash
tensorboard --logdir logs
```

Open http://localhost:6006 in your browser.

### Metrics
The trainer tracks:
- **Total Loss**: Combined detection + classification loss
- **Positive Accuracy**: % of face anchors correctly classified
- **Background Accuracy**: % of background anchors correctly classified
- **Mean IoU**: Average IoU for detected faces

## Output

### Checkpoints
Saved to `checkpoints/` directory:
- `BlazeFace_best.pth`: Best model (lowest validation loss)
- `BlazeFace_final.pth`: Final model after training
- `BlazeFace_epoch{N}.pth`: Periodic checkpoints (every 10 epochs by default)

### TensorBoard Logs
Saved to `logs/BlazeFace/` directory with:
- Training/validation losses
- Metrics (accuracy, IoU)
- Learning rate schedule

## Example Workflows

### Full Training Run
```bash
# Train for 500 epochs with default settings
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --epochs 500 \
    --batch-size 32 \
    --lr 1e-4
```

### With Focal Loss
```bash
# Use focal loss for class imbalance
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --use-focal-loss \
    --focal-alpha 0.25 \
    --focal-gamma 2.0
```

### With Pretrained Weights
```bash
# Initialize from MediaPipe weights
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --pretrained
```

### Debug Mode
```bash
# Quick test with small batch and 1 epoch
python train_blazeface.py \
    --csv-format \
    --train-data data/splits/train.csv \
    --val-data data/splits/val.csv \
    --data-root data/raw/blazeface \
    --batch-size 4 \
    --epochs 1 \
    --num-workers 0
```

## Dataset Statistics

Current dataset (WIDER Face format):
- **Total samples**: 5,881 face detections
- **Training**: 4,704 samples (80%)
- **Validation**: 1,176 samples (20%)
- **Image size**: Resized to 128x128
- **Anchors**: 896 per image (512 small + 384 large)

## Notes

- The model learns very quickly - you'll see IoU > 50% after just 1 epoch
- Loss values are weighted heavily (detection_weight=150, classification_weight=35)
- Training includes augmentation: brightness, saturation, horizontal flip
- The trainer uses hard negative mining (3:1 ratio) to handle class imbalance
