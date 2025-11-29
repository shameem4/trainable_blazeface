"""
Quick debug test for DINOv2 hybrid encoder.
Tests model initialization and forward pass.
"""

import torch
import sys
from pathlib import Path

# Fix Unicode encoding on Windows
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import EarVAE

def test_dino_encoder():
    """Test DINOv2 encoder initialization and forward pass."""
    print("="*60)
    print("Testing DINOv2 Hybrid Encoder")
    print("="*60)

    # Create model
    print("\n1. Initializing EarVAE with DINOv2 encoder...")
    latent_dim = 512
    image_size = 128

    try:
        model = EarVAE(latent_dim=latent_dim, image_size=image_size)
        print("[OK] Model initialized successfully")
    except Exception as e:
        print(f"[FAIL] Model initialization failed: {e}")
        return False

    # Print model info
    print(f"\n2. Model configuration:")
    print(f"   - Latent dim: {model.latent_dim}")
    print(f"   - Image size: {model.image_size}")
    print(f"   - Encoder type: {type(model.encoder).__name__}")
    print(f"   - Decoder type: {type(model.decoder).__name__}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n3. Parameter count:")
    print(f"   - Total: {total_params:,}")
    print(f"   - Trainable: {trainable_params:,}")
    print(f"   - Frozen (DINOv2): {frozen_params:,}")

    # Test forward pass
    print(f"\n4. Testing forward pass...")
    batch_size = 2

    try:
        # Create dummy input (batch_size, 3, 128, 128) in range [-1, 1]
        x = torch.randn(batch_size, 3, image_size, image_size)
        print(f"   - Input shape: {x.shape}")

        # Forward pass
        with torch.no_grad():
            recon, mu, logvar = model(x)

        print(f"   - Reconstruction shape: {recon.shape}")
        print(f"   - Latent mu shape: {mu.shape}")
        print(f"   - Latent logvar shape: {logvar.shape}")

        # Verify shapes
        assert recon.shape == (batch_size, 3, image_size, image_size), "Reconstruction shape mismatch"
        assert mu.shape == (batch_size, latent_dim), "Mu shape mismatch"
        assert logvar.shape == (batch_size, latent_dim), "Logvar shape mismatch"

        print("[OK] Forward pass successful!")

    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test sampling
    print(f"\n5. Testing latent space sampling...")
    try:
        with torch.no_grad():
            samples = model.sample(num_samples=4, device=x.device)

        print(f"   - Sample shape: {samples.shape}")
        assert samples.shape == (4, 3, image_size, image_size), "Sample shape mismatch"
        print("[OK] Sampling successful!")

    except Exception as e:
        print(f"[FAIL] Sampling failed: {e}")
        return False

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_dino_encoder()
    sys.exit(0 if success else 1)
