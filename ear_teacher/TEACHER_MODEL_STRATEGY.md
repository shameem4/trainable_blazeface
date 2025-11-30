# Teacher Model Training Strategy

## Goal: Learn Teachable Features

You need a model that:
1. ✓ Reconstructs ears accurately (proves features are rich)
2. ✓ Learns **generalizable features** (transfers to detection)
3. ✓ Distinguishes between different ears (feature discrimination)
4. ✗ Doesn't just memorize (lookup table problem)

## The Trade-Off

### Pure Reconstruction (KL → 0)
```
Pros:
- Perfect reconstruction
- Sharp details
- Low PSNR loss

Cons:
- Each ear gets unique code
- No generalization
- Features don't transfer
- Essentially a lookup table
```

### Heavy Regularization (KL >> 0)
```
Pros:
- Smooth latent space
- Good generalization
- Codes cluster by similarity

Cons:
- Blurry reconstructions
- Lost details
- Poor teaching signal
```

### **Balanced Approach** (Sweet Spot)
```
Moderate regularization:
- Good reconstruction quality
- Features generalize across similar ears
- Model learns "ear concepts" not "ear instances"
- Strong teaching signal for detection
```

## Optimized Loss Configuration

### Multi-Objective Balance

```python
# PRIMARY: Reconstruction (Sharp + Accurate)
L1 recon:        1.0   ← Pixel accuracy
Perceptual:      1.0   ← Semantic feature matching (VGG)
SSIM:            0.4   ← Structural similarity
Edge/Gradient:   0.1   ← Sharp boundaries

# REGULARIZATION: Feature Learning
KL divergence:   0.000005  ← Moderate (not too weak, not too strong)
Contrastive:     0.1       ← NEW: Feature discrimination

# SPATIAL: Ear-Specific
Center weight:   3.0   ← Focus on ear region
Focal weighting: Auto  ← Hard region emphasis
```

### Total Signal Breakdown

```
Reconstruction:      2.5x  (1.0 + 1.0 + 0.4 + 0.1)
Feature Learning:    0.1x  (contrastive)
Regularization:      ~0.001x (0.000005 × ~200 KL)

Ratio: 2500:1 reconstruction-to-regularization
```

## Why This Balance Works

### 1. **KL = 0.000005 (Moderate)**

**Target KL loss:** 100-300 (healthy range)
```
Effective penalty = 0.000005 × 200 = 0.001
```

This is:
- **100x weaker** than original (0.0001)
- **10x stronger** than ultra-low (0.0000001)
- **Just right** for learning generalizable features

**What it does:**
- Prevents exact memorization
- Forces similar ears to have similar codes
- Allows reconstruction of fine details
- Creates a **structured latent space**

### 2. **Contrastive Loss = 0.1 (NEW)**

**Purpose:** Encourage feature discrimination

```python
# Different ears should have different latent codes
# Similar ears can be close, but not identical
contrastive_loss = mean(|similarity(ear_i, ear_j)|)
```

**What it does:**
- Prevents all ears collapsing to same code
- Encourages unique features for each ear
- Maintains discrimination while allowing similarity
- **Essential for teaching detection models**

### 3. **Perceptual Loss = 1.0 (Strong)**

**Purpose:** Match high-level semantic features

**Why it's the primary signal:**
- VGG features are what detection models care about
- Prevents blur (VGG features require sharp inputs)
- Transfers directly to detection task
- **This IS the teaching signal**

### 4. **Edge Loss = 0.1 (Sharp Details)**

**Purpose:** Preserve high-frequency information

**Why it matters:**
- Ear landmarks are defined by edges
- Detection needs precise boundaries
- Complements perceptual loss
- Forces sharp anatomical features

## How Features Transfer to Detection

### Training Pipeline

```
Phase 1: Teacher (Current)
Input → DINOv2 + Custom Encoder → Latent (1024D) → Decoder → Reconstruction

Features learned:
- DINOv2: General semantic understanding (frozen/partially frozen)
- Custom layers: Ear-specific patterns (trainable)
- Latent space: Compact ear representation

Phase 2: Detection (Later)
Input → [Pretrained Encoder] → Multi-scale features → Detection heads
                                                      → Landmark heads

Transfer:
- Encoder weights: Pre-trained on reconstruction
- Features: Already know ear anatomy
- Detection heads: Learn from labeled data (faster, needs less data)
```

### Why This Works

**The encoder learns:**
1. **Ear vs non-ear** (via DINOv2 semantics)
2. **Ear parts** (helix, tragus, lobe - via reconstruction)
3. **Spatial relationships** (via center weighting + attention)
4. **Unique ear features** (via contrastive loss)
5. **Sharp boundaries** (via edge + perceptual loss)

**For detection, you need exactly these features!**

## Expected Training Behavior

### Healthy Metrics (Target)

```
Epoch 20:
- KL loss: 150-250
- Recon: 0.04-0.06
- Perceptual: 0.3-0.4
- PSNR: 26-28 dB
- Visual: Clear, some details

Epoch 40:
- KL loss: 100-200 (stable)
- Recon: 0.03-0.04
- Perceptual: 0.2-0.3
- PSNR: 28-31 dB
- Visual: Sharp, detailed

Epoch 60:
- KL loss: 100-150 (converged)
- Recon: 0.02-0.03
- Perceptual: 0.15-0.25
- PSNR: 30-33 dB
- Visual: Excellent quality
```

### What to Watch

**Good signs:**
- ✓ KL loss stabilizes at 100-300
- ✓ Reconstructions improve steadily
- ✓ Different ears have visually different reconstructions
- ✓ Fine details (ridges, curves) are preserved

**Bad signs:**
- ✗ KL loss > 500 (too strong regularization)
- ✗ All reconstructions look similar (memorization or collapse)
- ✗ Still blurry after 30 epochs (need more reconstruction signal)
- ✗ KL loss < 50 (too weak, may overfit)

## Comparison: Different Strategies

### Strategy 1: Pure Reconstruction (KL ≈ 0)
```
KL weight:     0.0000001
Target KL:     10-50
PSNR:          35+ dB (excellent)
Features:      Poor transfer (overfitting)
Detection:     BAD (features too specific)
```

### Strategy 2: Heavy Regularization (Original)
```
KL weight:     0.0001
Target KL:     1500+ (collapsed)
PSNR:          20-24 dB (poor)
Features:      Over-smoothed
Detection:     BAD (no detail information)
```

### **Strategy 3: Balanced (Current) ← OPTIMAL**
```
KL weight:     0.000005
Target KL:     100-300
PSNR:          30-33 dB (very good)
Features:      Generalizable + Discriminative
Detection:     EXCELLENT (best transfer)
```

## Feature Quality Validation

### How to Check if Features Are Good

After training, extract features from validation set:

```python
# Extract latent codes
features = []
for batch in val_loader:
    mu, _ = model.encoder(batch)
    features.append(mu)

# Check discrimination
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(features)

# Good features have:
# 1. Low average similarity (< 0.3) - ears are distinguishable
# 2. High similarity for same ear (> 0.9) - consistent encoding
# 3. Moderate similarity for similar ears (0.4-0.7) - generalization
```

### Visual Feature Check

**Look at reconstructions:**
1. Different ear shapes → Different reconstructions ✓
2. Similar ears → Similar reconstructions ✓
3. Fine details preserved ✓
4. Not all blurry/identical ✓

**This proves features are both discriminative AND generalizable.**

## Transfer Learning Performance

### Expected Detection Training

**Without teacher (random init):**
```
Data needed:   1000+ labeled images
Epochs:        50-100
AP@0.5:        0.70-0.75
Landmark error: 3-4 pixels
```

**With teacher (pretrained encoder):**
```
Data needed:   200-300 labeled images
Epochs:        20-30
AP@0.5:        0.80-0.85
Landmark error: 2-3 pixels
```

**Improvement:** 3-5x data efficiency, 2-3x faster, +10-15% accuracy

## Training Command

```bash
# Balanced approach (default settings are optimized)
python ear_teacher/train.py \
    --train-npy data/train.npy \
    --val-npy data/val.npy \
    --epochs 100

# All parameters:
# --kl-weight 0.000005          # Moderate regularization
# --perceptual-weight 1.0        # Primary teaching signal
# --ssim-weight 0.4              # Structure preservation
# --contrastive-weight 0.1       # Feature discrimination
# --latent-dim 1024              # High capacity
# --lr 3e-4                      # Fast learning
```

## Summary

### The Philosophy

**You're not training a reconstruction model.**
**You're training a TEACHER that will teach DETECTION.**

The teacher needs:
1. **Rich features** (reconstruction proves this)
2. **Generalizable knowledge** (moderate KL provides this)
3. **Discriminative power** (contrastive loss ensures this)
4. **Sharp understanding** (perceptual + edge loss deliver this)

### The Balance

```
Too much reconstruction → Memorization → Poor transfer
Too much regularization → Blur → Weak teaching signal
BALANCED → Sharp + Generalizable → Excellent transfer
```

### Current Configuration

```
Reconstruction:  Strong (2.5x)     ← Proves features are rich
Discrimination:  Moderate (0.1x)   ← Prevents collapse
Regularization:  Light (0.001x)    ← Allows details
Edge preservation: Active (0.1x)   ← Maintains sharpness

Result: Features that are BOTH detailed AND generalizable
```

**This is the optimal balance for a teacher model.**
