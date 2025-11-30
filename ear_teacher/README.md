# Ear Teacher Model

Teacher model for learning ear features via reconstruction, designed to transfer knowledge to detection and landmark models.

## Quick Start

### Training

```bash
cd ear_teacher
python train.py
```

All optimal settings are now defaults (Option 2: Sharp Reconstructions).

### Monitoring

```bash
cd ear_teacher
tensorboard --logdir logs
```

### Evaluation

```bash
cd ear_teacher
python evaluate.py
```

### Eigenears Visualization

```bash
cd ear_teacher
python eigenears/create_eigenears.py
```

Visualizes the principal components of the learned latent space to understand what features the model learned. See [eigenears/README.md](eigenears/README.md) for details.

## Directory Structure

```
ear_teacher/
├── train.py              # Training script
├── evaluate.py           # Model evaluation
├── model.py              # Model architectures (VAE, Encoder, Decoder)
├── lightning/
│   ├── module.py         # PyTorch Lightning training module
│   └── datamodule.py     # Data loading
├── eigenears/            # Eigenear visualization (PCA of latent space)
│   ├── create_eigenears.py  # Generate eigenear visualizations
│   └── README.md         # Eigenears documentation
├── checkpoints/          # Saved model checkpoints (created during training)
├── logs/                 # TensorBoard logs (created during training)
├── EVALUATION_REPORT.md          # Latest evaluation results
├── OPTION2_CHANGES.md            # Current configuration details
├── TEACHER_MODEL_STRATEGY.md     # Training philosophy & targets
├── FINAL_RECOMMENDATIONS.md      # Historical: Previous recommendations
├── DIAGNOSIS_FUZZY_RECONSTRUCTIONS.md  # Historical: Blur diagnosis
└── TRAINING_OPTIMIZATIONS.md     # Historical: DINOv2 optimizations
```

## Current Configuration (Option 2)

Optimized for **sharp, detailed reconstructions** + **discriminative features**:

```python
KL weight:          0.000001  # Ultra-low regularization
Perceptual weight:  1.5       # Strong semantic matching
SSIM weight:        0.6       # Structural preservation
Edge weight:        0.3       # Sharp boundaries (KEY)
Contrastive weight: 0.1       # Feature discrimination
Latent dim:         1024      # High capacity
Epochs:             60        # Faster convergence
```

**Expected Results:**
- PSNR: 27-30 dB (sharp details)
- Discrimination: 0.5-0.6 (good teaching capability)
- Training time: ~55 minutes

## Model Architecture

### Encoder: DINOv2 Hybrid

- **Base:** DINOv2-small (pretrained, partially frozen)
  - First 8 blocks: Frozen (general features)
  - Last 4 blocks: Trainable (ear-specific adaptation)
- **Custom layers:** Conv + Spatial Attention
- **Output:** Multi-scale features + 1024D latent code

### Decoder: Transposed Convolutions

- Upsampling from 1024D latent
- Progressive reconstruction: 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
- Skip connections for detail preservation

### VAE Training Objective

```
Total Loss = L1_recon + 1.5×Perceptual + 0.6×SSIM + 0.3×Edge + 0.1×Contrastive + 0.000001×KL
```

**Purpose:** Learn features rich enough to:
1. Reconstruct ear images sharply (proves feature quality)
2. Discriminate between different ears (enables detection)
3. Transfer to downstream tasks (detection, landmarks)

## Data Format

Training expects NPY files with structure:

```python
{
    'image_paths': [...],  # List of image file paths
    'bboxes': [...]        # Bounding boxes (not used for VAE training)
}
```

Images are automatically loaded, resized to 128×128, and normalized.

## Training Outputs

### Checkpoints

Saved in `ear_teacher/checkpoints/`:
- `last.ckpt` - Latest checkpoint
- `ear_vae-epoch=XXX-val/loss=Y.YYYY.ckpt` - Top 3 best models

### Logs

Saved in `ear_teacher/logs/ear_vae/version_X/`:
- `metrics.csv` - Training metrics
- `events.out.tfevents.*` - TensorBoard events
- `reconstructions/` - Visual reconstruction samples per epoch
- `hparams.yaml` - Hyperparameters used

## Evaluation Metrics

Run `python evaluate.py` to get:

1. **NaN/Inf Detection** - Inference stability
2. **Reconstruction Quality** - PSNR, SSIM, KL loss
3. **Feature Discrimination** - Pairwise similarity score
4. **Inference Stability** - Variance across runs

### Target Metrics

- ✅ PSNR ≥ 27 dB
- ✅ Discrimination score ≥ 0.5
- ✅ No NaN issues
- ✅ KL loss: 5-20 (minimal regularization)

## Using Trained Model for Detection

See `DETECTION_GUIDE.md` for full details. Quick overview:

```python
from model import EarDetector

# Load pretrained encoder from VAE checkpoint
detector = EarDetector.from_vae_checkpoint(
    checkpoint_path='checkpoints/last.ckpt',
    num_landmarks=17,
    freeze_encoder=False
)

# Fine-tune on labeled detection data
# ... training loop with bbox + landmark losses ...
```

## Hyperparameter Tuning

### If reconstructions are still blurry:

```bash
python train.py --edge-weight 0.5 --perceptual-weight 2.0
```

### If you need more regularization (features too specific):

```bash
python train.py --kl-weight 0.00001
```

### If training is unstable:

```bash
python train.py --lr 1e-4 --warmup-epochs 10
```

## Common Commands

### Resume from checkpoint:

```bash
python train.py --resume checkpoints/last.ckpt
```

### Change precision:

```bash
python train.py --precision 32          # Full precision
python train.py --precision bf16-mixed  # BFloat16 (A100/H100)
```

### Enable early stopping:

```bash
python train.py --early-stopping --patience 15
```

## Troubleshooting

### Out of memory:

```bash
python train.py --batch-size 16 --precision 16-mixed
```

### Training too slow:

- Reduce `--num-workers` if CPU-bound
- Increase `--num-workers` if I/O-bound
- Use `--precision 16-mixed` for faster training

### Poor reconstruction quality:

1. Check `logs/ear_vae/version_X/reconstructions/`
2. If blurry: Increase `--edge-weight` and `--perceptual-weight`
3. If wrong colors: Increase `--ssim-weight`
4. If overfitting: Increase `--kl-weight`

## Documentation

- **EVALUATION_REPORT.md** - Latest quantitative analysis (PSNR, discrimination, etc.)
- **TEACHER_MODEL_STRATEGY.md** - Why we use this architecture & loss configuration
- **OPTION2_CHANGES.md** - Details on current (optimized) configuration
- **DETECTION_GUIDE.md** - How to use trained encoder for detection/landmarks

## Citation & License

Based on:
- DINOv2: Meta AI's self-supervised vision transformer
- VAE: Variational Autoencoder framework
- PyTorch Lightning: Training framework

## Version History

- **v3 (Current):** Option 2 configuration - Sharp reconstructions (KL=0.000001, Edge=0.3)
- **v2:** Balanced configuration (KL=0.000005, Edge=0.1) - Good features but blurry
- **v1:** Original configuration (KL=0.0001) - Posterior collapse

---

**Status:** Ready for training with optimal configuration ✅

**Next:** Train model → Evaluate → Use for detection transfer learning
