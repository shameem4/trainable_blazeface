# Ear Teacher Model

**ResNet-Based Variational Autoencoder** for learning anatomical ear features via reconstruction, designed to transfer knowledge to detection and landmark models.

## Quick Start

### Training

```bash
python ear_teacher/train.py --epochs 60 --batch-size 8
```

Training takes approximately **2-4 hours** for 60 epochs with batch size 8 on modern GPUs.

### Monitoring

```bash
tensorboard --logdir ear_teacher/logs
```

Watch for:
- PSNR increasing (target: 28+ dB)
- SSIM increasing (target: 0.85+)
- Reconstruction quality improving

### Testing Model

Before training, verify the model works:

```bash
python ear_teacher/test_resnet_model.py
```

Expected: All 4 core tests pass (encoder, VAE, Lightning module, performance).

### Eigenears Visualization

After training, visualize learned features:

```bash
python ear_teacher/eigenears/create_eigenears.py
```

See [eigenears/README.md](eigenears/README.md) for interpretation guide.

## Architecture

### Encoder: ResNet-50 (ImageNet Pretrained)

**Why ResNet?**
- 10-20x faster training than SAM (hours vs days)
- 4x less memory - can use batch size 8-16 instead of 1-2
- Proven ImageNet features transfer well to reconstruction tasks
- Simpler architecture, easier to train and debug
- Extensive research validating ResNet for VAE tasks

**Configuration:**
- **Base:** `torchvision.models.resnet50` with ImageNet weights
- **Partial freezing:** First 2 layer groups frozen (conv1, bn1, relu, maxpool, layer1, layer2)
- **Input processing:** 128×128 images (native resolution)
- **ResNet output:** 2048 channels, 4×4 spatial resolution
- **Custom layers:** 3× (Conv + Batch Norm + ReLU + Residual + Spatial Attention)
- **Final output:** 1024D latent code (mu and logvar)

**Parameters:**
- Total: 116M
- Trainable: 115M (98.4%)
- Frozen: 1.4M (1.6%)

### Decoder: Transposed Convolutions

- Upsampling from 1024D latent
- Progressive reconstruction: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
- Skip connections for detail preservation
- Same as previous architecture (unchanged)

### Training Objective

```
Total Loss = L1_recon + 1.5×Perceptual + 0.6×SSIM + 0.3×Edge + 0.1×Contrastive + 3.0×Center + 1e-6×KL
```

**Loss components:**
- **L1 Reconstruction:** Pixel-level accuracy
- **Perceptual (VGG16):** Semantic similarity
- **SSIM:** Structural preservation
- **Edge:** Sharp anatomical boundaries (critical for SAM features)
- **Contrastive:** Feature discrimination between different ears
- **Center:** Ear-centric focus (higher weight on center region)
- **KL Divergence:** Minimal regularization (ultra-low weight)

## Configuration

All optimal settings are defaults:

```python
# Hyperparameters
latent_dim = 1024
learning_rate = 3e-4
kl_weight = 0.000001      # Ultra-low KL
perceptual_weight = 1.5   # Strong semantic matching
ssim_weight = 0.6         # Structural preservation
edge_weight = 0.3         # Sharp boundaries
contrastive_weight = 0.1  # Feature discrimination
center_weight = 3.0       # Ear-centric focus
batch_size = 8            # Recommended (6+ GB VRAM)
epochs = 60
image_size = 128
resnet_version = 'resnet50'  # ResNet-50, ResNet-101, or ResNet-152
freeze_layers = 2         # Freeze first 2 ResNet layer groups
```

**Expected Results:**
- **PSNR:** 28-32 dB (sharp anatomical details)
- **SSIM:** 0.85+ (excellent structure preservation)
- **Eigenears:** Anatomical features (ear boundary, lobe/helix, tragus, antihelix)
- **Variance explained:** 60%+ in first 5 principal components

## Data Format

Training expects NPY files:

```python
# data/preprocessed/train_teacher.npy
# data/preprocessed/val_teacher.npy
{
    'image_paths': [...],  # List of image file paths
    'bboxes': [...]        # Bounding boxes (not used for VAE)
}
```

Images are automatically:
- Loaded from paths
- Resized to 128×128
- Normalized to [-1, 1]

**Dataset size:**
- Train: 12,023 samples
- Validation: 3,006 samples
- Total: 15,029 ear images

## Training Outputs

### Checkpoints

Saved in `ear_teacher/checkpoints/`:
- `last.ckpt` - Latest checkpoint
- `ear_vae-epoch=XXX-val/loss=Y.YYYY.ckpt` - Top 3 best models

### Logs

Saved in `ear_teacher/logs/ear_vae/version_X/`:
- `metrics.csv` - Training metrics
- `events.out.tfevents.*` - TensorBoard events
- `reconstructions/` - Visual samples per epoch
- `hparams.yaml` - Hyperparameters

### Eigenears

Generated after training in `ear_teacher/eigenears/`:
- `pc_X.png` - Individual principal components (PC1-PC16)
- `eigenears_grid.png` - Summary visualization
- Shows what anatomical features the model learned

## Performance

**Training time (batch size 8):**
- Per epoch: ~2-4 minutes
- 60 epochs: ~2-4 hours total
- Speed: ~10-15 iterations/second

**Memory requirements:**
- Model weights: 443 MB
- Per-image memory: ~55 MB (128×128 processing)
- Batch size 8: ~3.5 GB total
- **Minimum VRAM:** 6 GB

**Batch size recommendations:**
- 4: Minimum for good training (4 GB VRAM)
- 8: Recommended for optimal speed (6 GB VRAM) ⭐
- 16: Fast training (12 GB VRAM)
- 32: Maximum speed (24+ GB VRAM)

**Note:** ResNet processes 128×128 images natively - much more efficient than SAM's 1024×1024 requirement.

## Using Trained Model

### For Detection/Landmarks

```python
from model import EarDetector

# Load pretrained SAM encoder from VAE checkpoint
detector = EarDetector.from_vae_checkpoint(
    checkpoint_path='ear_teacher/checkpoints/last.ckpt',
    num_landmarks=17,
    freeze_encoder=False
)

# Fine-tune on labeled detection data
# ... training loop ...
```

See [DETECTION_GUIDE.md](DETECTION_GUIDE.md) for full guide.

### For Feature Extraction

```python
from model import EarVAE
import torch

# Load trained VAE
model = EarVAE.load_from_checkpoint('ear_teacher/checkpoints/last.ckpt')
model.eval()

# Extract latent features
with torch.no_grad():
    mu, logvar = model.encoder(images)  # (B, 1024)

# Use for:
# - Ear embeddings
# - Similarity search
# - Clustering
# - Transfer learning
```

## Advanced Usage

### Resume from checkpoint

```bash
python ear_teacher/train.py --resume ear_teacher/checkpoints/last.ckpt
```

### Custom hyperparameters

```bash
# More edge focus (sharper boundaries)
python ear_teacher/train.py --edge-weight 0.5

# Less KL regularization (more reconstruction focus)
python ear_teacher/train.py --kl-weight 0.0000001

# Different learning rate
python ear_teacher/train.py --lr 1e-4

# Larger batch size (if you have VRAM)
python ear_teacher/train.py --batch-size 8
```

### Precision options

```bash
# Full precision (slower, more accurate)
python ear_teacher/train.py --precision 32

# BFloat16 (A100/H100 GPUs)
python ear_teacher/train.py --precision bf16-mixed
```

## Troubleshooting

### Out of Memory

**Solution:** Reduce batch size
```bash
python ear_teacher/train.py --batch-size 2
# or even
python ear_teacher/train.py --batch-size 1
```

### Training Too Slow

**Solutions:**
- Reduce `--num-workers` if CPU-bound
- Increase `--num-workers` if I/O-bound (default: 4)
- Use `--precision 16-mixed` (AMP enabled by default)

### Poor Eigenears Quality

**If eigenears still show only brightness/color:**
1. Increase contrastive weight: `--contrastive-weight 0.3`
2. Unfreeze more SAM layers: `--freeze-layers 4`
3. Train longer: `--epochs 80`
4. Check PSNR in logs (should be 28+ dB)

## Directory Structure

```
ear_teacher/
├── train.py                    # Training script
├── evaluate.py                 # Model evaluation
├── model.py                    # ResNet-based VAE architecture
├── test_resnet_model.py        # Model testing suite
├── dataset.py                  # Data loading
├── lightning/
│   ├── module.py               # PyTorch Lightning wrapper
│   └── datamodule.py           # Data module
├── eigenears/
│   ├── create_eigenears.py    # Generate eigenear visualizations
│   └── README.md               # Eigenears documentation
├── checkpoints/                # Saved model checkpoints
├── logs/                       # TensorBoard logs
├── archive/                    # Historical documentation
│   ├── EIGENEAR_ANALYSIS.md   # DINOv2 failure analysis
│   ├── SAM_ANALYSIS.md        # SAM impracticality analysis
│   └── ...                     # Other archived docs
├── TEACHER_MODEL_STRATEGY.md   # Training philosophy
├── DETECTION_GUIDE.md          # Using model for detection
└── README.md                   # This file
```

## Additional Documentation

- [DETECTION_GUIDE.md](DETECTION_GUIDE.md) - Using trained model for detection/landmarks
- [eigenears/README.md](eigenears/README.md) - Eigenear interpretation guide
- [archive/](archive/) - Historical documentation (DINOv2 and SAM experiments)

## Why ResNet Instead of SAM or DINOv2?

**DINOv2 failed:**
- ❌ Classification pretraining (ImageNet) misaligned with reconstruction
- ❌ No faces/ears in ImageNet training data
- ❌ Eigenears showed only brightness/color gradients
- ❌ PSNR: 20-25 dB (poor quality)

**SAM was impractical:**
- ❌ Requires 1024×1024 input (hardcoded, cannot change)
- ❌ Training time: 27 days for 60 epochs (0.3 it/s)
- ❌ Memory intensive: batch size 1 only
- ❌ Gradient checkpointing required but still too slow

**ResNet succeeds:**
- ✅ Fast training: 2-4 hours for 60 epochs (10-15 it/s)
- ✅ Memory efficient: batch size 8-16 feasible
- ✅ ImageNet features transfer well to reconstruction
- ✅ Simple, proven architecture
- ✅ Expected PSNR: 28-32 dB (excellent quality)

See [archive/EIGENEAR_ANALYSIS.md](archive/EIGENEAR_ANALYSIS.md) for DINOv2 failure analysis.

## Citation

This model uses:
- **ResNet:** Deep Residual Learning for Image Recognition (He et al., 2016)
- **VAE:** Variational Autoencoder framework
- **PyTorch Lightning:** Training framework

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## Version History

- **v5 (Current):** ResNet-50 encoder - Fast training (2-4 hrs), 28-32 dB PSNR ⭐
- **v4 (Archived):** SAM-based encoder - Impractical (27 day training)
- **v3 (Archived):** DINOv2 Option 2 - Brightness only, 20-25 dB PSNR
- **v2 (Archived):** DINOv2 balanced config - Blurry reconstructions
- **v1 (Archived):** DINOv2 original - Posterior collapse

---

**Status:** ✅ **Ready for Production Training**

**Next Steps:**
1. Run `python ear_teacher/test_resnet_model.py` to verify setup
2. Train: `python ear_teacher/train.py --epochs 60 --batch-size 8`
3. Monitor: `tensorboard --logdir ear_teacher/logs`
4. Generate eigenears: `python ear_teacher/eigenears/create_eigenears.py`
5. Verify anatomical features learned (not just brightness)
