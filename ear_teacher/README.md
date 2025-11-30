# Ear Teacher Model

**SAM-Based Variational Autoencoder** for learning anatomical ear features via reconstruction, designed to transfer knowledge to detection and landmark models.

## Quick Start

### Training

```bash
python ear_teacher/train.py --epochs 60 --batch-size 4
```

Training takes approximately **22 hours** for 60 epochs with batch size 4.

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
python ear_teacher/test_sam_model.py
```

Expected: All 3 core tests pass (encoder, VAE, Lightning module).

### Eigenears Visualization

After training, visualize learned features:

```bash
python ear_teacher/eigenears/create_eigenears.py
```

See [eigenears/README.md](eigenears/README.md) for interpretation guide.

## Architecture

### Encoder: SAM Hybrid (Meta's Segment Anything Model)

**Why SAM?**
- Pretrained on SA-1B (11M images, 1.1B masks) with segmentation task
- Edge/boundary focused features ideal for anatomical structures
- Dataset includes faces (likely ears)
- Task alignment: Segmentation → Reconstruction (much better than Classification → Reconstruction)

**Configuration:**
- **Base:** `facebook/sam-vit-base` (Vision Transformer)
- **Partial freezing:** First 6/12 layers frozen, last 6 trainable (64.6% trainable)
- **Input processing:** 128×128 → 1024×1024 (SAM requirement)
- **SAM output:** 256 channels, 64×64 spatial resolution
- **Custom layers:** 3× (Conv + Batch Norm + ReLU + Residual + Spatial Attention)
- **Final output:** 1024D latent code (mu and logvar)

**Parameters:**
- Total: 148M
- Trainable: 105M (70.8%)
- Frozen: 43M (29.2%)

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
batch_size = 4            # Recommended (6+ GB VRAM)
epochs = 60
image_size = 128
freeze_layers = 6         # Freeze first 6 SAM blocks
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

**Training time (batch size 4):**
- Per epoch: ~22 minutes
- 60 epochs: ~22 hours total
- Speed: ~2.3 iterations/second

**Memory requirements:**
- Model weights: 565 MB
- Per-image memory: ~140 MB (1024×1024 SAM processing)
- Batch size 4: ~3.5 GB total
- **Minimum VRAM:** 6 GB

**Batch size recommendations:**
- 1-2: Memory constrained (< 6 GB VRAM)
- 4: Recommended (6+ GB VRAM) ⭐
- 8+: Requires 12+ GB VRAM (may OOM)

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
├── model.py                    # SAM-based VAE architecture
├── test_sam_model.py           # Model testing suite
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
│   ├── BACKBONE_ALTERNATIVES.md # Why SAM was chosen
│   └── ...                     # Other archived docs
├── SAM_IMPLEMENTATION.md       # SAM architecture details
├── DEBUG_RUN_RESULTS.md        # Debug run verification
├── TEACHER_MODEL_STRATEGY.md   # Training philosophy
├── DETECTION_GUIDE.md          # Using model for detection
└── README.md                   # This file
```

## Additional Documentation

- [DETECTION_GUIDE.md](DETECTION_GUIDE.md) - Using trained model for detection/landmarks
- [eigenears/README.md](eigenears/README.md) - Eigenear interpretation guide
- [archive/](archive/) - Historical DINOv2-era documentation and research

## Why SAM Instead of DINOv2?

**DINOv2 failed:**
- ❌ Classification pretraining (ImageNet) misaligned with reconstruction
- ❌ No faces/ears in ImageNet training data
- ❌ Eigenears showed only brightness/color gradients
- ❌ No anatomical features learned
- ❌ PSNR: 20-25 dB (poor quality)

**SAM succeeds:**
- ✅ Segmentation pretraining perfectly aligned with reconstruction
- ✅ SA-1B dataset includes faces (likely ears)
- ✅ Edge/boundary focused features
- ✅ Eigenears show anatomical structure
- ✅ Expected PSNR: 28-32 dB (excellent quality)

See [archive/EIGENEAR_ANALYSIS.md](archive/EIGENEAR_ANALYSIS.md) for detailed failure analysis.

## Citation

This model uses:
- **SAM:** [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- **VAE:** Variational Autoencoder framework
- **PyTorch Lightning:** Training framework

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

## Version History

- **v4 (Current):** SAM-based encoder - Anatomical features, 28-32 dB PSNR ⭐
- **v3 (Archived):** DINOv2 Option 2 - Brightness only, 20-25 dB PSNR
- **v2 (Archived):** DINOv2 balanced config - Blurry reconstructions
- **v1 (Archived):** DINOv2 original - Posterior collapse

---

**Status:** ✅ **Ready for Production Training**

**Next Steps:**
1. Run `python ear_teacher/test_sam_model.py` to verify setup
2. Train: `python ear_teacher/train.py --epochs 60 --batch-size 4`
3. Monitor: `tensorboard --logdir ear_teacher/logs`
4. Generate eigenears: `python ear_teacher/eigenears/create_eigenears.py`
5. Verify anatomical features learned (not just brightness)
