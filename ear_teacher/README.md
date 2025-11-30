# Ear Teacher - VAE with Spatial Attention

A Convolutional Variational Autoencoder (VAE) with spatial attention mechanisms for learning intricate details of human ear images. Built with **PyTorch Lightning**, **torchmetrics**, **torchvision.ops**, and **albumentations** for robust and professional training.

This teacher module is designed to learn rich representations that will later be used to train ear detector and landmarker networks.

## Architecture

### Model Components

1. **Convolutional VAE**: Deep encoder-decoder architecture with 4 downsampling/upsampling layers
2. **Spatial Attention**: Channel-spatial attention mechanisms to focus on important ear features
3. **Squeeze-Excitation**: torchvision.ops SqueezeExcitation modules at encoder/decoder bottlenecks
4. **Latent Space**: 256-dimensional latent representation for capturing ear variations

### Network Details

- **Input**: RGB or grayscale ear images (default: 3 channels, 256x256)
- **Encoder**: 4 attention blocks with progressive channel increase (64 → 128 → 256 → 512)
  - Squeeze-Excitation at bottleneck for enhanced channel relationships
- **Latent**: 256-dimensional continuous latent space with KL divergence annealing
- **Decoder**: 4 attention blocks with progressive channel decrease (512 → 256 → 128 → 64)
  - Squeeze-Excitation at input for smooth decoding
- **Output**: Reconstructed ear images with sigmoid activation

### Attention Mechanism

The model uses a combined channel-spatial attention mechanism:
- **Channel Attention**: Uses both average and max pooling to learn channel-wise relationships
- **Spatial Attention**: Focuses on important spatial locations in feature maps
- **Squeeze-Excitation**: Additional channel recalibration via torchvision.ops
- Applied at multiple scales throughout the encoder-decoder architecture

## Features

✅ **PyTorch Lightning**: Professional training loop with callbacks and logging
✅ **Torchmetrics**: SSIM, PSNR, and loss metrics for comprehensive evaluation
✅ **Torchvision.ops**: Squeeze-Excitation modules for enhanced feature learning
✅ **Albumentations**: Rich data augmentation pipeline for robustness
✅ **Cyclic KL Annealing**: State-of-the-art annealing strategy (Fu et al., 2019)
✅ **Normalized Losses**: Batch-size and image-size invariant loss computation
✅ **Automatic Checkpointing**: Best model based on loss and SSIM
✅ **Early Stopping**: Prevents overfitting with patience-based stopping
✅ **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
✅ **TensorBoard Integration**: Real-time monitoring with image logging
✅ **Mixed Precision**: Support for fp16/bf16 training
✅ **Modular Design**: Clean, maintainable codebase with proper separation of concerns

## Data Format

The model expects preprocessed data in NumPy format:
- Training data: `data/preprocessed/train_teacher.npy`
- Validation data: `data/preprocessed/val_teacher.npy`

Data should be in shape `(N, H, W, C)` or `(N, C, H, W)` and will be automatically normalized to [0, 1].

## Installation

Install dependencies:

```bash
pip install -r ear_teacher/requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `pytorch-lightning>=2.0.0`
- `torchmetrics>=1.0.0`
- `albumentations>=1.3.0`
- `opencv-python>=4.8.0`

## Training

### Basic Usage

The training script can be run in two ways from the repository root:

**Method 1: As a module (recommended)**

```bash
python -m ear_teacher.train
```

**Method 2: Standalone script**

```bash
python ear_teacher/train.py
```

### Training Arguments

```bash
python -m ear_teacher.train \
    --train_data data/preprocessed/train_teacher.npy \
    --val_data data/preprocessed/val_teacher.npy \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --latent_dim 256 \
    --in_channels 3 \
    --image_size 256 \
    --accelerator gpu \
    --devices 1 \
    --precision 16
```

### Key Parameters

**Data:**
- `--train_data`: Path to training .npy file
- `--val_data`: Path to validation .npy file

**Model:**
- `--in_channels`: Input channels (1=grayscale, 3=RGB, default: 3)
- `--latent_dim`: Latent space dimensionality (default: 256)
- `--base_channels`: Base channel count (default: 64)
- `--image_size`: Input image size (default: 256)

**Training:**
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for AdamW (default: 1e-5)
- `--kld_weight`: Final KL divergence weight (default: 0.00025)

**KLD Annealing:**
- `--kld_anneal_strategy`: Annealing strategy - 'cyclic', 'linear', or 'monotonic' (default: cyclic)
- `--kld_anneal_cycles`: Number of cycles for cyclic annealing (default: 4)
- `--kld_anneal_ratio`: Ratio of increasing phase per cycle, 0.0-1.0 (default: 0.5)
- `--kld_anneal_start`: Starting weight for KLD (default: 0.0)
- `--kld_anneal_end`: Ending weight multiplier (default: 1.0)

**Hardware:**
- `--accelerator`: Device type (auto, gpu, cpu, tpu, default: auto)
- `--devices`: Number of devices (default: 1)
- `--precision`: Training precision (32, 16, bf16, default: 32)
- `--num_workers`: Data loading workers (default: 4)

**Checkpointing:**
- `--checkpoint_dir`: Checkpoint directory (default: `ear_teacher/checkpoints`)
- `--log_dir`: TensorBoard log directory (default: `ear_teacher/logs`)
- `--resume`: Path to checkpoint to resume from (default: auto-detect `last.ckpt`)

**Other:**
- `--fast_dev_run`: Quick test with 1 batch
- `--seed`: Random seed (default: 42)

## Data Augmentation

The training pipeline uses **albumentations** for comprehensive augmentation:

### Training Augmentations
- **Geometric**: HorizontalFlip, ShiftScaleRotate, Perspective, ElasticTransform
- **Color**: ColorJitter, RandomBrightnessContrast, HueSaturationValue
- **Noise/Blur**: GaussNoise, GaussianBlur, MotionBlur
- **Occlusion**: CoarseDropout
- **Environmental**: RandomShadow, RandomFog

### Validation
- No augmentation, only resize and normalization

## Checkpointing & Resuming

### Automatic Checkpoint Saving

Lightning automatically saves:
- **Best model (loss)**: `best-{epoch}-{val/loss}.ckpt` - Best validation loss
- **Best model (SSIM)**: `best-ssim-{epoch}-{val/ssim}.ckpt` - Best SSIM score
- **Last checkpoint**: `last.ckpt` - Most recent epoch (for resuming)

### Resume Training

The training script automatically resumes from `last.ckpt` if it exists:

```bash
# Automatically resumes if last.ckpt exists
python -m ear_teacher.train
```

To resume from a specific checkpoint:

```bash
python -m ear_teacher.train --resume ear_teacher/checkpoints/best-epoch=099-val/loss=0.001234.ckpt
```

## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir ear_teacher/logs
```

**Logged Metrics:**
- **Training**: loss, recon_loss, kld, kld_weight, lr
- **Validation**: loss, recon_loss, kld, SSIM, PSNR
- **Images**: Sample reconstructions every epoch

### Progress Bar

Rich progress bar shows real-time metrics:
- Training loss, validation loss
- SSIM and PSNR scores
- Current learning rate
- ETA and time per epoch

## Loss Function

The VAE loss combines two components with proper normalization and annealing:

### Normalized Loss Computation

Both losses are normalized to ensure consistent scaling regardless of batch size or image dimensions:

1. **Reconstruction Loss**: MSE normalized per pixel
   - `recon_loss = mean(MSE(reconstruction, target))`
   - Scales consistently across different image sizes

2. **KL Divergence**: Normalized per latent dimension
   - `KLD = mean(-0.5 * sum(1 + log(σ²) - μ² - σ²))`
   - Averaged over batch for stability

**Total Loss:**
```
Total Loss = Reconstruction Loss + (current_kld_weight × KL Divergence)
```

**Benefits of Normalization:**
- ✅ Consistent loss values across different batch sizes
- ✅ Invariant to image resolution changes
- ✅ Easier hyperparameter tuning
- ✅ Stable training dynamics

### KL Divergence Annealing

The model supports three annealing strategies to prevent posterior collapse and improve training:

#### 1. Cyclic Annealing (Default, Recommended)

Based on "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing" (Fu et al., 2019).

**How it works:**
- The KLD weight follows a repeating cyclic pattern
- Each cycle: increases linearly from 0 to target weight, then holds constant
- Multiple cycles throughout training

**Benefits:**
- **Prevents posterior collapse**: Periodic resets allow model to explore latent space
- **Better reconstructions**: Balances reconstruction quality and latent regularization
- **Improved training dynamics**: Multiple opportunities to learn meaningful representations
- **State-of-the-art performance**: Proven effective in VAE literature

**Parameters:**
- `--kld_anneal_cycles 4`: Number of complete cycles (default: 4)
- `--kld_anneal_ratio 0.5`: Portion of each cycle spent increasing (0.5 = 50% increase, 50% constant)

**Example - 200 epochs with 4 cycles:**
```bash
python -m ear_teacher.train \
    --kld_anneal_strategy cyclic \
    --kld_anneal_cycles 4 \
    --kld_anneal_ratio 0.5 \
    --kld_weight 0.00025
```

With 4 cycles over 200 epochs:
- Each cycle = 50 epochs
- Epochs 0-25: KLD weight increases 0 → 0.00025
- Epochs 25-50: KLD weight stays at 0.00025
- Epochs 50-75: KLD weight increases 0 → 0.00025 (cycle 2)
- ...and so on

**Visualization:**
```
KLD Weight
   ^
   |     ╱‾‾‾‾╲     ╱‾‾‾‾╲     ╱‾‾‾‾╲     ╱‾‾‾‾
   |    ╱      ╲   ╱      ╲   ╱      ╲   ╱
   |   ╱        ╲ ╱        ╲ ╱        ╲ ╱
   |  ╱          ╳          ╳          ╳
   | ╱          ╱ ╲        ╱ ╲        ╱ ╲
   |╱__________╱   ╲______╱   ╲______╱   ╲______
   +------------------------------------------------> Epochs
     Cycle 1      Cycle 2    Cycle 3    Cycle 4
```

#### 2. Monotonic Annealing

Smooth increase using cosine schedule - starts slow, accelerates in middle, slows at end.

**Example:**
```bash
python -m ear_teacher.train \
    --kld_anneal_strategy monotonic \
    --kld_weight 0.00025
```

#### 3. Linear Annealing

Simple linear increase from start to end weight.

**Example:**
```bash
python -m ear_teacher.train \
    --kld_anneal_strategy linear \
    --kld_weight 0.00025
```

#### Advanced Configuration

Control the annealing range with start/end multipliers:

```bash
# Start at 0%, end at 100% of target weight
python -m ear_teacher.train \
    --kld_anneal_start 0.0 \
    --kld_anneal_end 1.0 \
    --kld_weight 0.00025

# Start at 10%, end at 80% of target weight
python -m ear_teacher.train \
    --kld_anneal_start 0.1 \
    --kld_anneal_end 0.8 \
    --kld_weight 0.00025
```

**Why Cyclic Annealing Works:**

1. **Exploration**: Low KLD weight periods allow model to prioritize reconstruction
2. **Regularization**: High KLD weight periods enforce meaningful latent structure
3. **Avoids Local Minima**: Periodic resets help escape poor solutions
4. **Multiple Refinements**: Each cycle refines both reconstruction and latent space

## Metrics

### Training Metrics
- **Loss**: Total VAE loss (recon + weighted KLD)
- **Recon Loss**: MSE reconstruction error
- **KLD**: KL divergence value
- **KLD Weight**: Current annealing weight
- **LR**: Learning rate

### Validation Metrics
- **Loss**: Total VAE loss
- **Recon Loss**: MSE reconstruction error
- **KLD**: KL divergence value
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **PSNR**: Peak Signal-to-Noise Ratio (dB, higher is better)

## Callbacks

### ModelCheckpoint
- Saves best model based on validation loss
- Saves best model based on SSIM
- Saves last checkpoint for resuming

### EarlyStopping
- Monitors validation loss
- Patience of 30 epochs
- Prevents overfitting

### LearningRateMonitor
- Logs learning rate to TensorBoard
- Tracks ReduceLROnPlateau adjustments

### RichProgressBar
- Beautiful CLI progress visualization
- Real-time metric updates

## Output

### Checkpoints

Saved in `ear_teacher/checkpoints/`:
- `best-{epoch}-{val/loss}.ckpt`: Best validation loss
- `best-ssim-{epoch}-{val/ssim}.ckpt`: Best SSIM
- `last.ckpt`: Most recent (for resuming)

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- LR scheduler state dict
- Hyperparameters
- Current epoch
- All metric states

### Logs

TensorBoard logs in `ear_teacher/logs/ear_vae/` containing:
- Scalar metrics (loss, SSIM, PSNR, etc.)
- Sample reconstructions as images
- Learning rate curves
- Hyperparameters

## Inference

### Loading Trained Model

```python
from ear_teacher import EarVAELightning

# Load from checkpoint
model = EarVAELightning.load_from_checkpoint('ear_teacher/checkpoints/best-*.ckpt')
model.eval()

# Encode images to latent space
latent = model.encode(images)

# Decode latent vectors to images
reconstructions = model.decode(latent)

# Reconstruct images directly
reconstructions = model.reconstruct(images)

# Sample random ear images
samples = model.sample(num_samples=8)
```

## Future Use

The trained VAE will be used to:
1. **Ear Detection**: Generate features for training ear detectors
2. **Landmark Detection**: Provide rich representations for landmark prediction
3. **Feature Extraction**: Extract robust ear features for downstream tasks
4. **Data Augmentation**: Generate synthetic ear images for training

## Tips for Best Results

1. **Start with annealing**: Use KLD annealing for first 50 epochs
2. **Monitor SSIM**: Better perceptual quality indicator than MSE
3. **Use mixed precision**: `--precision 16` for faster training
4. **Adjust batch size**: Larger batches (64-128) often improve stability
5. **Check reconstructions**: Validate model quality visually in TensorBoard
6. **Early stopping**: Let patience=30 prevent overfitting naturally

## Troubleshooting

### Posterior Collapse
- Symptoms: KLD → 0, poor reconstructions
- Solution: Increase `kld_anneal_epochs`, decrease `kld_weight`

### Blurry Reconstructions
- Symptoms: MSE is low but images lack detail
- Solution: Increase `kld_weight`, train longer, check SSIM

### Training Instability
- Symptoms: Loss spikes, NaN values
- Solution: Reduce learning rate, increase gradient clipping, check data normalization

### Out of Memory
- Solution: Reduce `batch_size`, use smaller `base_channels`, enable `--precision 16`

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## Model Testing

Test model architecture:

```bash
python -m ear_teacher.model
```

Quick training test:

```bash
python -m ear_teacher.train --fast_dev_run
```
