"""
Test script to verify SAM-based model initialization and forward pass.

This script tests:
1. Model initialization with SAM backbone
2. Pretrained weights loading
3. Forward pass with dummy data
4. Feature extraction
5. Memory usage and speed

Run this before training to ensure everything works.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import EarVAE, SAMHybridEncoder
from lightning.module import EarVAELightning


def test_sam_encoder():
    """Test SAM encoder initialization and forward pass."""
    print("="*60)
    print("TEST 1: SAM Encoder")
    print("="*60)

    # Initialize encoder
    print("\n1. Initializing SAM encoder...")
    encoder = SAMHybridEncoder(
        latent_dim=1024,
        image_size=128,
        freeze_layers=6,
        use_pretrained=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n2. Parameter count:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters:    {frozen_params:,}")
    print(f"   Trainable ratio:      {trainable_params/total_params*100:.1f}%")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 128, 128)

    with torch.no_grad():
        mu, logvar = encoder(dummy_input)

    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   mu shape:     {mu.shape}")
    print(f"   logvar shape: {logvar.shape}")

    # Check for NaNs
    if torch.isnan(mu).any() or torch.isnan(logvar).any():
        print("   [FAIL] NaN detected in output!")
        return False
    else:
        print("   [OK] No NaN in output")

    # Test feature extraction
    print("\n4. Testing feature extraction...")
    with torch.no_grad():
        features = encoder.extract_features(dummy_input)

    print(f"   Feature pyramid:")
    for key, feat in features.items():
        print(f"     {key:6s}: {feat.shape}")

    print("\n[OK] SAM encoder test passed!")
    return True


def test_vae_model():
    """Test complete VAE model."""
    print("\n" + "="*60)
    print("TEST 2: Complete VAE Model")
    print("="*60)

    # Initialize model
    print("\n1. Initializing SAM-based VAE...")
    model = EarVAE(
        latent_dim=1024,
        image_size=128,
        freeze_layers=6,
        use_pretrained=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n2. Full model parameter count:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n3. Testing VAE forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 128, 128)

    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)

    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Recon shape:  {recon.shape}")
    print(f"   mu shape:     {mu.shape}")
    print(f"   logvar shape: {logvar.shape}")

    # Check output range
    print(f"\n4. Output statistics:")
    print(f"   Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
    print(f"   mu mean:              {mu.mean():.3f}")
    print(f"   mu std:               {mu.std():.3f}")
    print(f"   logvar mean:          {logvar.mean():.3f}")

    # Check for NaNs
    if torch.isnan(recon).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
        print("   [FAIL] NaN detected in output!")
        return False
    else:
        print("   [OK] No NaN in output")

    # Test sampling
    print("\n5. Testing sampling...")
    with torch.no_grad():
        samples = model.sample(num_samples=4, device='cpu')

    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

    print("\n[OK] VAE model test passed!")
    return True


def test_lightning_module():
    """Test PyTorch Lightning module."""
    print("\n" + "="*60)
    print("TEST 3: Lightning Module")
    print("="*60)

    # Initialize Lightning module
    print("\n1. Initializing Lightning module...")
    lightning_model = EarVAELightning(
        latent_dim=1024,
        learning_rate=3e-4,
        kl_weight=0.000001,
        perceptual_weight=1.5,
        ssim_weight=0.6,
        edge_weight=0.3,
        contrastive_weight=0.1,
        center_weight=3.0,
        recon_loss_type='l1',
        image_size=128
    )

    print(f"   Hyperparameters:")
    for key, value in lightning_model.hparams.items():
        print(f"     {key:20s}: {value}")

    # Test training step
    print("\n2. Testing training step...")
    batch_size = 4
    dummy_batch = torch.randn(batch_size, 3, 128, 128)

    with torch.no_grad():
        # Compute losses (without gradients for testing)
        recon, mu, logvar = lightning_model.model(dummy_batch)

    print(f"   Batch shape:        {dummy_batch.shape}")
    print(f"   Reconstruction:     {recon.shape}")

    # Check for NaNs
    if torch.isnan(recon).any():
        print("   [FAIL] NaN in reconstruction!")
        return False
    else:
        print("   [OK] No NaN in reconstruction")

    print("\n[OK] Lightning module test passed!")
    return True


def test_memory_and_speed():
    """Test memory usage and forward pass speed."""
    print("\n" + "="*60)
    print("TEST 4: Memory and Speed")
    print("="*60)

    model = EarVAE(latent_dim=1024, image_size=128, freeze_layers=6)
    model.eval()

    batch_sizes = [1, 4, 8, 16]

    print("\nBatch size | Forward time (ms) | Memory (MB)")
    print("-" * 50)

    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 128, 128)

        # Warmup
        with torch.no_grad():
            _ = model(dummy_input)

        # Measure time
        import time
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        elapsed = (time.time() - start) * 1000

        # Estimate memory (rough)
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
        total_memory = param_memory + buffer_memory

        print(f"{bs:10d} | {elapsed:16.1f} | {total_memory:11.1f}")

    print("\n[OK] Performance test completed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SAM-BASED VAE MODEL TEST SUITE")
    print("="*60)
    print("\nThis will test:")
    print("  1. SAM encoder initialization and forward pass")
    print("  2. Complete VAE model")
    print("  3. Lightning module")
    print("  4. Memory usage and speed")
    print("\n" + "="*60)

    try:
        # Run tests
        test1 = test_sam_encoder()
        test2 = test_vae_model()
        test3 = test_lightning_module()
        test4 = test_memory_and_speed()

        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  SAM Encoder:      {'[PASS]' if test1 else '[FAIL]'}")
        print(f"  VAE Model:        {'[PASS]' if test2 else '[FAIL]'}")
        print(f"  Lightning Module: {'[PASS]' if test3 else '[FAIL]'}")
        print(f"  Performance:      {'[PASS]' if test4 else '[FAIL]'}")

        if all([test1, test2, test3, test4]):
            print("\n[SUCCESS] All tests passed! Ready to train.")
            print("\nNext steps:")
            print("  1. Clear old checkpoints (optional):")
            print("     rm -rf checkpoints/* logs/*")
            print("  2. Start training:")
            print("     cd ear_teacher")
            print("     python train.py")
            print("  3. Monitor progress:")
            print("     tensorboard --logdir logs")
            return 0
        else:
            print("\n[FAILURE] Some tests failed. Please fix errors before training.")
            return 1

    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
