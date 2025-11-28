# Ear Teacher - Convolutional VAE for Ear Representation Learning

A deep convolutional Variational Autoencoder (VAE) designed to learn rich
representations of human ears from cropped ear images.

## Architecture

### Model: Deep Convolutional VAE

- **Encoder**: 5-layer CNN with residual blocks
  - Input: (3, 128, 128)
  - Progressive downsampling: 128 → 64 → 32 → 16 → 8 → 4
  - Channels: 64 → 128 → 256 → 512 → 512
  - Output: Latent space (512-dimensional by default)

- **Decoder**: 5-layer transposed CNN with residual blocks
  - Mirrors encoder architecture
  - Progressive upsampling: 4 → 8 → 16 → 32 → 64 → 128
  - Output: Reconstructed image (3, 128, 128)

### Loss Function

Multi-component loss for high-quality reconstructions:

- **Reconstruction Loss**: MSE or L1 between original and reconstructed images
- **KL Divergence**: Regularizes latent space to be normally distributed
- **Perceptual Loss**: VGG16-based feature matching for semantic similarity
- **SSIM Loss**: Structural similarity for perceptual quality

Total Loss = Recon + λ_KL × KL + λ_percept × Perceptual + λ_SSIM × (1 - SSIM)

## Data Augmentation

Comprehensive augmentations via Albumentations:

### Geometric Transformations

- **Scale jitter**: ±30% random scaling
- **Translation jitter**: ±10% random shifts
- **Rotation**: ±30° random rotations
- **Random cropping**: 70-100% of image area
- **Horizontal flip**: 50% probability

### Photometric Augmentations

- **Color jitter**: Brightness, contrast, saturation, hue (±30%)
- **RGB shift**: Random channel shifts
- **Grayscale conversion**: 10% probability
- **Random gamma**: Gamma correction (80-120)
- **Brightness/contrast**: Additional ±20% adjustments

### Quality Degradations

- **Blur**: Gaussian, motion, or median blur
- **Gaussian noise**: Variable intensity (10-50)
- **JPEG compression**: Quality 60-100

### Synthetic Occlusions

- **Coarse dropout**: 3-8 random rectangular holes (up to 20% size)
- **Grid dropout**: 30% grid-based occlusion

## Installation

```bash
# Install required packages
pip install torch torchvision pytorch-lightning
pip install albumentations torchmetrics
pip install tensorboard
```

## Usage

### 1. Prepare Data

Process raw annotations into NPY metadata files:

```bash
cd earmesh
python -m shared.data_processing.data_processor --teacher
```

This creates:

- `data/preprocessed/train_teacher.npy` (~1-5MB)
- `data/preprocessed/val_teacher.npy` (~500KB)

Each NPY file contains:

- `image_paths`: Paths to images on disk
- `bboxes`: Bounding boxes for cropping ear regions

### 2. Train Model

Basic training:

```bash
cd ear_teacher
python train.py
```

Custom configuration:

```bash
python train.py \
  --batch-size 64 \
  --epochs 300 \
  --lr 2e-4 \
  --latent-dim 512 \
  --image-size 128 \
  --kl-weight 0.0001 \
  --perceptual-weight 0.5 \
  --ssim-weight 0.1 \
  --gpus 1 \
  --num-workers 8 \
  --early-stopping \
  --patience 30
```

### 3. Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/ear_teacher
```

Metrics logged:

- Training/validation loss (total and per-component)
- SSIM and PSNR
- Learning rate
- Reconstruction visualizations

### 4. Resume Training

```bash
python train.py --resume checkpoints/ear_teacher/last.ckpt
```

## Project Structure

```text
ear_teacher/
├── __init__.py           # Package exports
├── model.py              # VAE architecture (encoder, decoder, losses)
├── dataset.py            # Dataset with augmentations
├── train.py              # Training script
├── lightning/            # PyTorch Lightning wrappers
│   ├── __init__.py
│   ├── module.py         # LightningModule with training loop
│   └── datamodule.py     # LightningDataModule
└── README.md             # This file
```

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `latent_dim` | 512 | Latent space dimensionality |
| `image_size` | 128 | Input image size (square) |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-4 | Initial learning rate |
| `kl_weight` | 0.0001 | KL divergence weight |
| `perceptual_weight` | 0.5 | Perceptual loss weight |
| `ssim_weight` | 0.1 | SSIM loss weight |
| `warmup_epochs` | 5 | LR warmup duration |

### Loss Weight Tuning

- **Higher KL weight** (e.g., 0.001): Better latent space structure,
  but may reduce reconstruction quality
- **Lower KL weight** (e.g., 0.00001): Better reconstructions,
  but latent space may be less smooth
- **Higher perceptual weight** (e.g., 1.0): More semantically
  accurate reconstructions
- **Higher SSIM weight** (e.g., 0.3): Better perceptual quality,
  but may sacrifice pixel-level accuracy

## Advanced Usage

### Inference

```python
from ear_teacher.lightning import EarVAELightning

# Load trained model
model = EarVAELightning.load_from_checkpoint(
  'checkpoints/ear_vae-epoch=100.ckpt'
)
model.eval()
model.cuda()

# Encode ear image to latent representation
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Load and preprocess image
image = Image.open('ear.jpg').convert('RGB')
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
image_tensor = transform(image=np.array(image))['image'].unsqueeze(0).cuda()

# Get latent representation
with torch.no_grad():
    latent = model.model.encode(image_tensor)  # (1, 512)

# Reconstruct image
with torch.no_grad():
    recon = model.model.decode(latent)  # (1, 3, 128, 128)
```

### Sample Random Ears

```python
# Generate random ear samples from latent space
num_samples = 16
samples = model.model.sample(num_samples, device='cuda')

# Visualize
import torchvision
grid = torchvision.utils.make_grid((samples + 1) / 2, nrow=4)
torchvision.utils.save_image(grid, 'random_ears.png')
```

## Performance

Expected metrics after convergence:

- **SSIM**: 0.85-0.92 (higher is better)
- **PSNR**: 22-28 dB (higher is better)
- **Reconstruction Loss**: 0.01-0.05 (lower is better)
- **KL Divergence**: 50-200 (depends on weight)

Training time (approximate):

- **GPU**: ~2-4 hours for 200 epochs (RTX 3090, batch_size=64)
- **CPU**: Not recommended (very slow)

## Troubleshooting

### Poor Reconstructions

- Decrease KL weight
- Increase perceptual weight
- Increase SSIM weight
- Train for more epochs

### Blurry Outputs

- Use L1 reconstruction loss instead of MSE
- Increase perceptual weight
- Decrease SSIM weight

### Mode Collapse (All reconstructions look similar)

- Increase KL weight
- Reduce learning rate
- Add more augmentation

### Out of Memory

- Reduce batch size
- Reduce image size
- Reduce latent dimension
- Use mixed precision (--precision 16-mixed)

## Citation

If you use this code, please cite:

```bibtex
@software{ear_teacher_vae,
  title={Ear Teacher: Convolutional VAE for Ear Representation Learning},
  author={Your Name},
  year={2025}
}
```
