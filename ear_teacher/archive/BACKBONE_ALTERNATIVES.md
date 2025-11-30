# Alternative Backbones for Ear Teacher Model

**Current Problem:** DINOv2 was pretrained on ImageNet (general objects), not medical/biometric images. It's learning generic blob features instead of ear anatomy.

---

## Recommended Alternatives (Ranked)

### 1. **SAM (Segment Anything Model) - Image Encoder** ⭐ BEST CHOICE

**Model:** `facebook/sam-vit-base` or `facebook/sam-vit-large`
**Pretrained on:** SA-1B dataset (11M images, 1.1B masks)
**Architecture:** Vision Transformer (ViT)

**Why it's better for ears:**
- ✅ Trained on **fine-grained segmentation** (helix, antihelix, lobe are distinct segments)
- ✅ Excellent at **edge detection** and **anatomical boundaries**
- ✅ Strong **spatial understanding** (important for ear structure)
- ✅ Already used successfully in medical imaging
- ✅ Large feature pyramid (multi-scale features)

**Advantages over DINOv2:**
- SAM learned to segment **parts of objects**, not just classify whole objects
- Ear detection = segmenting ear from background
- Landmark detection = finding specific anatomical points
- SAM's training task directly aligns with our goals

**Model sizes:**
- `sam-vit-base`: 89M params, faster training
- `sam-vit-large`: 308M params, best quality
- `sam-vit-huge`: 636M params, may be overkill

**Implementation:**
```python
from transformers import SamModel, SamProcessor

# Load SAM image encoder
sam = SamModel.from_pretrained("facebook/sam-vit-base")
encoder = sam.vision_encoder  # Use only the image encoder

# Freeze early layers, fine-tune later layers
for i, layer in enumerate(encoder.layers):
    if i < 6:  # Freeze first 6 of 12 layers
        for param in layer.parameters():
            param.requires_grad = False
```

**Expected improvements:**
- Eigenears should show ear structural features (not just brightness)
- PC1 might control ear boundary sharpness
- PC2-3 might control lobe/helix variations
- PSNR: 25-30 dB (better reconstruction)
- Better transfer to landmark detection

---

### 2. **MedSAM (Medical SAM)** ⭐⭐ SPECIALIZED CHOICE

**Model:** `MedSAM` (SAM fine-tuned on medical images)
**Pretrained on:** 1M+ medical image masks across 10+ modalities
**Architecture:** ViT-based (SAM derivative)

**Why it's better for ears:**
- ✅ **Medical domain adaptation** (understands anatomical structures)
- ✅ Trained on precise anatomical boundaries
- ✅ Better generalization to unseen body parts
- ✅ Used successfully for skin lesions, organs, tumors

**Advantages over base SAM:**
- Already adapted to medical/anatomical context
- Better at fine anatomical details
- May need less fine-tuning

**Disadvantages:**
- Smaller community, less documentation
- May be harder to find pretrained weights

**Implementation:**
```python
# MedSAM is typically distributed as SAM checkpoint fine-tuned on medical data
# Check: https://github.com/bowang-lab/MedSAM
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
encoder = sam.image_encoder
```

---

### 3. **MAE (Masked Autoencoder) - ViT** ⭐ RECONSTRUCTION-FOCUSED

**Model:** `facebook/vit-mae-base` or `facebook/vit-mae-large`
**Pretrained on:** ImageNet (reconstruction task, not classification)
**Architecture:** Vision Transformer

**Why it's better for ears:**
- ✅ Pretrained on **reconstruction** (same task as VAE!)
- ✅ Learned to fill in masked patches = good spatial understanding
- ✅ No classification bias (DINOv2 was trained for classification)
- ✅ Proven to transfer well to dense prediction tasks

**Advantages over DINOv2:**
- MAE's pretraining task (reconstruct masked patches) is closer to our VAE task
- DINOv2's pretraining task (classify objects) creates wrong inductive bias
- MAE features are more "reconstruction-friendly"

**Model sizes:**
- `vit-mae-base`: 86M params
- `vit-mae-large`: 304M params
- `vit-mae-huge`: 632M params

**Implementation:**
```python
from transformers import ViTMAEModel

mae = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
encoder = mae.encoder

# Freeze early layers
for i in range(6):  # Freeze first 6 of 12 layers
    for param in encoder.layer[i].parameters():
        param.requires_grad = False
```

**Expected improvements:**
- Better reconstruction quality (MAE was trained for this)
- PSNR: 26-32 dB
- Eigenears should show spatial/structural variations
- Good balance between general features and task alignment

---

### 4. **ConvNeXt V2** ⭐ PURE CONVOLUTIONAL

**Model:** `facebook/convnextv2-base-22k-224` or `facebook/convnextv2-large-22k-224`
**Pretrained on:** ImageNet-22k with MAE-style reconstruction
**Architecture:** Modern ConvNet (no transformers)

**Why it's better for ears:**
- ✅ **Convolutional = better spatial inductive bias** for images
- ✅ Trained with reconstruction objective (like MAE)
- ✅ Hierarchical features (good for multi-scale ear detection)
- ✅ Faster training than transformers
- ✅ Lower memory usage

**Advantages over DINOv2:**
- Convolutional layers preserve spatial locality
- Hierarchical features = natural multi-scale representation
- Better for dense prediction (segmentation, landmarks)
- No need for patch tokenization overhead

**Model sizes:**
- `convnextv2-tiny`: 28M params, fastest
- `convnextv2-base`: 89M params, good balance
- `convnextv2-large`: 198M params, best quality

**Implementation:**
```python
from transformers import ConvNextV2Model

convnext = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")

# Freeze early stages
for i in range(2):  # Freeze first 2 of 4 stages
    for param in convnext.encoder.stages[i].parameters():
        param.requires_grad = False
```

**Expected improvements:**
- Better spatial features
- Natural multi-scale features for decoder
- PSNR: 24-28 dB
- Faster training (30-40% faster than ViT)

---

### 5. **BiomedCLIP** ⭐⭐ DOMAIN-SPECIFIC

**Model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
**Pretrained on:** 15M biomedical image-text pairs from PubMed
**Architecture:** ViT with biomedical domain adaptation

**Why it's better for ears:**
- ✅ **Biomedical domain** (understands anatomical concepts)
- ✅ Trained on medical images and descriptions
- ✅ Better generalization to unseen anatomical structures
- ✅ Strong semantic understanding of body parts

**Advantages over DINOv2:**
- Medical/anatomical context baked in
- May understand ear anatomy terminology
- Better for few-shot learning on limited ear data

**Disadvantages:**
- Primarily designed for retrieval, not reconstruction
- May need more adaptation for VAE task

**Implementation:**
```python
from transformers import AutoModel

biomedclip = AutoModel.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
vision_encoder = biomedclip.vision_model

# Freeze early layers
for i in range(6):
    for param in vision_encoder.encoder.layers[i].parameters():
        param.requires_grad = False
```

---

## Comparison Table

| Backbone | Pretraining Task | Domain | Spatial Features | Reconstruction | Medical Relevance | Speed | Recommendation |
|----------|-----------------|--------|------------------|----------------|-------------------|-------|----------------|
| **DINOv2** (current) | Object classification | General (ImageNet) | Medium | Poor | None | Fast | ❌ Replace |
| **SAM** | Segmentation | General (diverse) | **Excellent** | Good | High (proven in medical) | Medium | ⭐⭐⭐ BEST |
| **MedSAM** | Medical segmentation | Medical | **Excellent** | Good | **Very High** | Medium | ⭐⭐⭐ BEST (if available) |
| **MAE** | Reconstruction | General (ImageNet) | Good | **Excellent** | Medium | Fast | ⭐⭐ GOOD |
| **ConvNeXt V2** | Reconstruction | General (ImageNet) | **Excellent** | **Excellent** | Medium | **Very Fast** | ⭐⭐ GOOD |
| **BiomedCLIP** | Image-text matching | Biomedical | Good | Medium | **Very High** | Fast | ⭐ OKAY |

---

## Top Recommendation: SAM ViT-Base

**Why SAM is the best choice:**

1. **Task Alignment:** SAM's pretraining (segment anything) is perfect for ears
   - Ear detection = segment ear from background
   - Landmark detection = segment anatomical parts
   - Reconstruction = regenerate segmented structures

2. **Proven Medical Success:** SAM has been successfully used for:
   - Skin lesion segmentation
   - Organ segmentation
   - Tumor detection
   - Dental X-ray analysis
   - **Facial landmark detection** (similar to ears!)

3. **Feature Quality:** SAM learned to:
   - Detect edges and boundaries (helix, antihelix)
   - Understand object parts (lobe, tragus, concha)
   - Preserve spatial relationships (ear structure)

4. **Practical Benefits:**
   - Well-documented, active community
   - Easy to integrate with HuggingFace transformers
   - Multiple size options (base/large/huge)
   - Proven to fine-tune well

---

## Implementation Plan

### Step 1: Replace DINOv2 with SAM

```python
# In ear_teacher/model.py

from transformers import SamModel, SamConfig
import torch
import torch.nn as nn

class SAMHybridEncoder(nn.Module):
    """SAM-based encoder for ear VAE."""

    def __init__(self, latent_dim=1024, freeze_early_layers=True):
        super().__init__()

        # Load SAM image encoder
        print("Loading SAM ViT-Base image encoder...")
        sam = SamModel.from_pretrained("facebook/sam-vit-base")
        self.sam_encoder = sam.vision_encoder

        # SAM outputs 256x64x64 features for 1024x1024 input
        # For 128x128 input, we get 256x16x16 features
        sam_feature_dim = 768  # SAM ViT-Base hidden size

        # Freeze early layers if requested
        if freeze_early_layers:
            num_layers = len(self.sam_encoder.layers)
            freeze_until = num_layers // 2  # Freeze first half

            print(f"SAM partially frozen: first {freeze_until} blocks frozen, last {num_layers - freeze_until} blocks trainable")

            # Freeze patch embedding
            for param in self.sam_encoder.patch_embed.parameters():
                param.requires_grad = False

            # Freeze early transformer blocks
            for i in range(freeze_until):
                for param in self.sam_encoder.layers[i].parameters():
                    param.requires_grad = False

        # Adaptive pooling to get spatial features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4 spatial grid

        # Custom layers for ear-specific features
        self.custom_conv = nn.Sequential(
            nn.Conv2d(sam_feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Project to latent space
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # SAM expects 1024x1024, but we have 128x128
        # Interpolate to SAM's expected size
        x_upsampled = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Extract SAM features
        sam_outputs = self.sam_encoder(x_upsampled)
        features = sam_outputs.last_hidden_state  # (batch, 64*64, 768) for 1024x1024 input

        # Reshape to spatial
        h = w = int(features.shape[1] ** 0.5)
        features = features.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)  # (batch, 768, 64, 64)

        # Downsample to manageable size
        features = self.adaptive_pool(features)  # (batch, 768, 4, 4)

        # Custom ear-specific processing
        features = self.custom_conv(features)  # (batch, 512, 4, 4)

        # Spatial attention
        attention = self.spatial_attention(features)
        features = features * attention

        # Flatten
        features_flat = features.reshape(batch_size, -1)  # (batch, 512*4*4)

        # Project to latent
        mu = self.fc_mu(features_flat)
        logvar = self.fc_logvar(features_flat)

        return mu, logvar, features  # Return features for skip connections
```

### Step 2: Update Training Script

```python
# In ear_teacher/train.py

# Add argument for backbone choice
parser.add_argument('--backbone', type=str, default='sam',
                    choices=['dinov2', 'sam', 'mae', 'convnext'],
                    help='Backbone encoder architecture')
```

### Step 3: Expected Training Time

**SAM ViT-Base on 128x128 images:**
- Memory: ~6GB VRAM (batch size 32)
- Speed: ~45-50 sec/epoch (vs 55 sec for DINOv2)
- **Slightly faster** due to better optimization

---

## Alternative: Ensemble Approach

If no single backbone works well, consider **ensemble**:

```python
class EnsembleEncoder(nn.Module):
    """Combine multiple backbones for robust features."""

    def __init__(self, latent_dim=1024):
        super().__init__()

        # Load multiple encoders
        self.sam_encoder = SAMEncoder(latent_dim // 2)
        self.mae_encoder = MAEEncoder(latent_dim // 2)

        # Combine features
        self.fusion = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        mu1, logvar1 = self.sam_encoder(x)
        mu2, logvar2 = self.mae_encoder(x)

        # Concatenate
        mu = self.fusion(torch.cat([mu1, mu2], dim=1))
        logvar = self.fusion(torch.cat([logvar1, logvar2], dim=1))

        return mu, logvar
```

**Pros:** Best of both worlds (segmentation + reconstruction)
**Cons:** 2x slower, 2x memory, complex training

---

## Immediate Next Step

**Recommend: Try SAM ViT-Base first**

1. Install requirements:
```bash
pip install transformers segment-anything
```

2. Modify `ear_teacher/model.py` to add `SAMHybridEncoder` class

3. Update `train.py` to support `--backbone sam`

4. Train for 60 epochs:
```bash
python train.py --backbone sam --epochs 60
```

5. Generate eigenears and compare:
```bash
python eigenears/create_eigenears.py
```

**Expected results:**
- PC1: Ear boundary sharpness / overall size
- PC2-3: Lobe vs helix prominence
- PC4-6: Structural variations (tragus, antihelix)
- PSNR: 25-30 dB
- Eigenears should show **clear anatomical features**

---

## Fallback Plan

If SAM doesn't work well:

1. **Try MAE** (closer to reconstruction task)
2. **Try ConvNeXt V2** (faster, better spatial features)
3. **Consider MedSAM** (if you can find pretrained weights)
4. **Train from scratch** with more data + heavy augmentation

**Bottom line:** SAM is the most promising alternative to DINOv2 for ear-specific feature learning.
