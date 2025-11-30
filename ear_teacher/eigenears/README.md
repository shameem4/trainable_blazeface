# Eigenears: Principal Component Analysis of Ear Latent Space

Visualizes the dominant modes of variation in ear appearance learned by the teacher model.

## What are Eigenears?

Similar to "eigenfaces" in face recognition, **eigenears** are the principal components of the learned latent space. They reveal:

- **What the model learned:** The main axes of variation in ear appearance
- **Interpretable features:** Each component often corresponds to anatomical variations (size, shape, orientation, texture)
- **Dimensionality:** How many components are needed to represent most ear variation

## Quick Start

```bash
cd ear_teacher
python eigenears/create_eigenears.py
```

This will:
1. Load your trained teacher model (`checkpoints/last.ckpt`)
2. Extract latent codes from validation images
3. Compute PCA on the latent space
4. Generate visualizations showing each principal component

## Output Files

Generated in `ear_teacher/eigenears/`:

### Individual Component Visualizations
- `eigenear_pc01.png` to `eigenear_pc16.png` - Each shows one principal component
  - Images range from -3σ to +3σ along that component
  - Title shows percentage of variance explained
  - Shows what anatomical feature that component controls

### Summary Visualization
- `eigenears_summary.png` - Grid of all 16 principal components
  - Each subplot is a heatmap of the component vector
  - Red/blue indicates positive/negative weights in latent space

### PCA Model
- `pca_model.pkl` - Saved PCA model for later analysis
  - Can be loaded to project new ears into eigenear space
  - Contains: PCA object, mean code, explained variance

## What to Look For

### Principal Component 1 (PC1)
- Usually explains most variance (e.g., 15-25%)
- Often corresponds to **overall ear size** or **brightness**
- Moving along PC1 should show ears getting larger/smaller or lighter/darker

### Principal Component 2-3 (PC2-PC3)
- Next largest sources of variation
- Often correspond to **shape variations**:
  - Ear width vs height ratio
  - Lobe size
  - Helix curvature

### Later Components (PC4-PC16)
- Smaller, more subtle variations
- May capture:
  - Texture details
  - Specific anatomical features (tragus shape, antihelix prominence)
  - Lighting/pose variations
  - Individual ear "styles"

### Cumulative Variance
- Check how much variance the top 5-10 components explain
- Target: 80%+ with first 10 components indicates good compression
- If variance is spread evenly, may indicate the model hasn't learned strong structure

## Interpretation Examples

**Good eigenear (interpretable):**
```
PC1: -3σ = small ear with pointed helix
      0σ = average ear
     +3σ = large ear with rounded helix
```
- Clear, smooth transitions
- Anatomically plausible at all points
- Single interpretable feature varying

**Poor eigenear (entangled):**
```
PC5: -3σ = realistic ear
      0σ = realistic ear
     +3σ = distorted/artifact-heavy ear
```
- Extreme values produce artifacts
- Multiple features changing simultaneously
- May indicate model hasn't fully disentangled features

## Advanced Usage

### Visualize More/Fewer Components

```python
# In create_eigenears.py, change:
n_components = 32  # Default is 16
```

### Adjust Visualization Range

```python
# Show wider range (-5σ to +5σ)
visualize_eigenear_component(
    model, pca, mean_code, component_idx,
    std_range=5.0,  # Default is 3.0
    steps=11,       # Default is 9
    device=device
)
```

### Project New Ear into Eigenear Space

```python
import pickle
import numpy as np

# Load PCA model
with open('eigenears/pca_model.pkl', 'rb') as f:
    pca_data = pickle.load(f)

pca = pca_data['pca']
mean_code = pca_data['mean_code']

# Get latent code for new ear
# latent_code = model.encode(new_ear_image)  # (1024,)

# Project into eigenear space
centered = latent_code - mean_code
eigenear_coords = pca.transform(centered.reshape(1, -1))

# eigenear_coords now contains the coordinates in PC space
# Shape: (1, 16) for 16 components
```

### Reconstruct from Eigenear Coordinates

```python
# Reconstruct latent code from first 5 PCs only
n_components_to_use = 5
reconstructed_latent = mean_code + pca.inverse_transform(
    eigenear_coords[:, :n_components_to_use]
)

# Decode to image
reconstructed_image = model.decode(reconstructed_latent)
```

## Troubleshooting

### "No clear pattern in PC visualizations"
- Model may need more training
- Or: Latent space is highly entangled
- Try: Increase contrastive weight in training

### "Artifacts at extreme σ values"
- Normal for extreme values (±3σ are rare in distribution)
- If artifacts appear at ±1σ, model may be underfitting
- Solution: Check PSNR and reconstruction quality first

### "All PCs look similar"
- May indicate posterior collapse (KL too high)
- Check: KL loss should be 2-20, not 100+
- Solution: Lower kl_weight in training

### "First PC explains <5% variance"
- Unusual - suggests latent space is very high-dimensional
- Check: Are you using the right checkpoint?
- Compare: Total variance explained by first 10 PCs

## Citation

Inspired by:
- **Eigenfaces:** Turk & Pentland (1991)
- **Disentangled Representations:** β-VAE (Higgins et al., 2017)
- **Principal Component Analysis:** Pearson (1901)

## Example Use Cases

1. **Model Debugging:** Check if model learned meaningful features
2. **Feature Selection:** Identify which PCs to use for downstream tasks
3. **Data Augmentation:** Generate new ears by sampling in PC space
4. **Compression:** Represent ears with 16 numbers instead of 1024
5. **Similarity Search:** Use PC coordinates for fast ear matching

---

**Status:** Ready to use with any trained teacher model ✅

**Next:** Run the script to visualize what your model learned!
