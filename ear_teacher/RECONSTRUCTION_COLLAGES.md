# Reconstruction Collages

## Overview

The training pipeline now automatically generates and saves reconstruction collages at the end of each validation epoch. These collages provide visual feedback on the model's reconstruction quality during training.

## What Gets Saved

- **Location**: `ear_teacher/logs/ear_vae/version_X/reconstructions/`
- **Filename Pattern**: `epoch_XXX.png`
- **Content**: Side-by-side comparison of 10 validation images (input on left, reconstruction on right)

## During Training

Reconstruction collages are automatically saved:
1. At the end of each validation epoch
2. Using the first batch of validation data
3. In the same directory where `metrics.csv` is saved
4. Stored in a `reconstructions/` subdirectory

## Implementation Details

The feature is implemented in the `EarVAELightning` module:

- `validation_step()`: Captures the first validation batch
- `on_validation_epoch_end()`: Triggers collage generation
- `_save_reconstruction_collage()`: Creates and saves the collage image

### File Structure

```
ear_teacher/logs/ear_vae/
└── version_0/
    ├── metrics.csv
    ├── hparams.yaml
    └── reconstructions/
        ├── epoch_000.png
        ├── epoch_001.png
        ├── epoch_002.png
        └── ...
```

## Standalone Collage Generation

You can also generate reconstruction collages independently using the `create_collage.py` script:

```bash
# From project root
python ear_teacher/create_collage.py

# With custom options
python ear_teacher/create_collage.py \
    --checkpoint ear_teacher/checkpoints/best-epoch=XXX.ckpt \
    --num_samples 20 \
    --output custom_collage.png
```

### Options

- `--checkpoint`: Path to model checkpoint (default: last.ckpt)
- `--val_data`: Path to validation data (default: data/preprocessed/val_teacher.npy)
- `--output`: Output path for collage (default: ear_teacher/logs/reconstruction_collage.png)
- `--num_samples`: Number of samples to show (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)

## Benefits

1. **Visual Progress Tracking**: See reconstruction quality improve over epochs
2. **Quick Quality Assessment**: Identify issues without analyzing metrics
3. **Model Comparison**: Compare different training runs visually
4. **Debugging**: Spot mode collapse, artifacts, or other training issues early

## Notes

- Collages use denormalized images (converted from [-1, 1] to [0, 1] range)
- Images are resized to 256x256 before reconstruction
- Only 10 samples are saved per epoch to keep file sizes reasonable
- Collages are saved at 100 DPI for good quality while managing disk space
