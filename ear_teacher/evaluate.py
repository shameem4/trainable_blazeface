"""
Comprehensive evaluation of trained teacher model.

Tests:
1. NaN/Inf detection in outputs
2. Reconstruction quality metrics
3. Feature discrimination quality
4. Inference stability
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from lightning.module import EarVAELightning
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def load_validation_data():
    """Load validation data from preprocessed files."""
    # Try different possible locations
    possible_paths = [
        'data/preprocessed/val_teacher.npy',
        '../data/preprocessed/val_teacher.npy',
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"Loading data from: {path}")
            data = np.load(path, allow_pickle=True)

            # Handle different data formats
            if data.dtype == object:
                data_dict = data.item()
                if 'image_paths' in data_dict:
                    # Load images from paths
                    from PIL import Image
                    import torchvision.transforms as T

                    transform = T.Compose([
                        T.Resize((128, 128)),
                        T.ToTensor(),
                    ])

                    images = []
                    paths = data_dict['image_paths'][:100]  # Limit to 100 for speed
                    print(f"Loading {len(paths)} images...")

                    for img_path in paths:
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = transform(img)
                            images.append(img_tensor)
                        except Exception as e:
                            print(f"Warning: Failed to load {img_path}: {e}")

                    if images:
                        return torch.stack(images)
            else:
                return torch.from_numpy(data).float()

    raise FileNotFoundError("Could not find validation data")


def check_for_nans(tensor, name):
    """Check if tensor contains NaN or Inf values."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        print(f"  [FAIL] {name}: NaN={has_nan}, Inf={has_inf}")
        return True
    else:
        print(f"  [OK] {name}: Clean (no NaN/Inf)")
        return False


def evaluate_reconstruction_quality(model, data_loader, device):
    """Evaluate reconstruction metrics."""
    print("\n" + "="*60)
    print("1. RECONSTRUCTION QUALITY")
    print("="*60)

    all_kl = []
    all_recon = []
    all_perceptual = []
    all_psnr = []
    all_ssim = []
    all_contrastive = []

    nan_batches = 0
    total_batches = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(data_loader):
            batch = batch.to(device)
            total_batches += 1

            # Forward pass
            recon, mu, logvar = model.model(batch)

            # Check for NaNs
            batch_has_nan = False
            if check_for_nans(recon, f"Batch {batch_idx} reconstruction"):
                batch_has_nan = True
            if check_for_nans(mu, f"Batch {batch_idx} latent mu"):
                batch_has_nan = True
            if check_for_nans(logvar, f"Batch {batch_idx} latent logvar"):
                batch_has_nan = True

            if batch_has_nan:
                nan_batches += 1
                print(f"  [WARN] Batch {batch_idx} contains NaN/Inf values!")
                continue

            # Compute losses
            try:
                losses = model._compute_losses(batch, recon, mu, logvar)

                all_kl.append(losses['kl_loss'].item())
                all_recon.append(losses['recon_loss'].item())
                all_perceptual.append(losses['perceptual'].item())
                all_psnr.append(losses['psnr'].item())
                all_ssim.append(losses['ssim'].item())

                if 'contrastive_loss' in losses and not torch.isnan(losses['contrastive_loss']):
                    all_contrastive.append(losses['contrastive_loss'].item())

            except Exception as e:
                print(f"  [WARN] Error computing losses for batch {batch_idx}: {e}")
                nan_batches += 1

    # Summary statistics
    print(f"\n{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 68)

    metrics = {
        'KL Loss': all_kl,
        'Recon Loss (L1)': all_recon,
        'Perceptual': all_perceptual,
        'PSNR (dB)': all_psnr,
        'SSIM': all_ssim,
    }

    if all_contrastive:
        metrics['Contrastive'] = all_contrastive

    for name, values in metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{name:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

    print(f"\n{'NaN/Inf Detection:':<20} {nan_batches}/{total_batches} batches affected")

    return {
        'nan_ratio': nan_batches / total_batches,
        'kl_mean': np.mean(all_kl) if all_kl else float('nan'),
        'psnr_mean': np.mean(all_psnr) if all_psnr else float('nan'),
        'ssim_mean': np.mean(all_ssim) if all_ssim else float('nan'),
    }


def evaluate_feature_discrimination(model, data_loader, device):
    """Evaluate whether features are discriminative enough to teach detection."""
    print("\n" + "="*60)
    print("2. FEATURE DISCRIMINATION (Teaching Capability)")
    print("="*60)

    all_mu = []
    all_images = []

    model.eval()
    with torch.no_grad():
        for batch, in data_loader:
            batch = batch.to(device)
            _, mu, _ = model.model(batch)

            if not torch.isnan(mu).any():
                all_mu.append(mu.cpu().numpy())
                all_images.append(batch.cpu())

    if not all_mu:
        print("[FAIL] No valid features extracted (all NaN)")
        return {'discrimination_score': 0.0}

    # Concatenate all features
    features = np.vstack(all_mu)
    print(f"Extracted features shape: {features.shape}")

    # Compute pairwise similarities
    similarities = cosine_similarity(features)

    # Mask diagonal (self-similarity)
    np.fill_diagonal(similarities, 0)

    # Statistics
    off_diagonal = similarities[np.triu_indices_from(similarities, k=1)]

    print(f"\nPairwise Similarity Statistics:")
    print(f"  Mean similarity:    {np.mean(np.abs(off_diagonal)):.4f}")
    print(f"  Std similarity:     {np.std(off_diagonal):.4f}")
    print(f"  Max similarity:     {np.max(np.abs(off_diagonal)):.4f}")
    print(f"  Min similarity:     {np.min(np.abs(off_diagonal)):.4f}")

    # Discrimination quality
    mean_sim = np.mean(np.abs(off_diagonal))

    print(f"\n{'Assessment:':<20}")
    if mean_sim < 0.3:
        print(f"  [OK] EXCELLENT discrimination (mean similarity: {mean_sim:.4f})")
        print(f"    Features are highly discriminative - good for teaching!")
    elif mean_sim < 0.5:
        print(f"  [OK] GOOD discrimination (mean similarity: {mean_sim:.4f})")
        print(f"    Features should work well for detection")
    elif mean_sim < 0.7:
        print(f"  [WARN] MODERATE discrimination (mean similarity: {mean_sim:.4f})")
        print(f"    Features may need improvement")
    else:
        print(f"  [FAIL] POOR discrimination (mean similarity: {mean_sim:.4f})")
        print(f"    Features too similar - may not teach well")

    # Check for mode collapse
    std_per_dim = np.std(features, axis=0)
    dead_dims = np.sum(std_per_dim < 0.01)

    print(f"\nLatent Space Health:")
    print(f"  Active dimensions:  {features.shape[1] - dead_dims}/{features.shape[1]}")
    print(f"  Dead dimensions:    {dead_dims}")

    if dead_dims > features.shape[1] * 0.1:
        print(f"  [WARN] Warning: >10% of latent dimensions are inactive")
    else:
        print(f"  [OK] Latent space is healthy")

    return {
        'discrimination_score': 1.0 - mean_sim,
        'mean_similarity': mean_sim,
        'dead_dimensions': dead_dims,
    }


def evaluate_inference_stability(model, data_loader, device):
    """Test inference stability with multiple runs."""
    print("\n" + "="*60)
    print("3. INFERENCE STABILITY")
    print("="*60)

    model.eval()

    # Get first batch
    batch = next(iter(data_loader))[0].to(device)

    print(f"Running inference 10 times on same batch...")

    reconstructions = []
    latents = []

    with torch.no_grad():
        for i in range(10):
            recon, mu, logvar = model.model(batch)

            if torch.isnan(recon).any() or torch.isnan(mu).any():
                print(f"  [WARN] Run {i}: Contains NaN")
            else:
                reconstructions.append(recon.cpu())
                latents.append(mu.cpu())

    if len(reconstructions) < 2:
        print("[FAIL] Not enough valid runs to assess stability")
        return {'stability_score': 0.0}

    # Check consistency
    recon_variance = torch.var(torch.stack(reconstructions), dim=0).mean().item()
    latent_variance = torch.var(torch.stack(latents), dim=0).mean().item()

    print(f"\nVariance across runs:")
    print(f"  Reconstruction variance: {recon_variance:.6f}")
    print(f"  Latent variance:         {latent_variance:.6f}")

    if recon_variance < 1e-10 and latent_variance < 1e-10:
        print(f"  [OK] EXCELLENT stability (deterministic)")
    elif recon_variance < 1e-6:
        print(f"  [OK] GOOD stability")
    else:
        print(f"  [WARN] Some variance detected (check for dropout/batchnorm issues)")

    return {
        'stability_score': 1.0 if recon_variance < 1e-6 else 0.5,
        'recon_variance': recon_variance,
        'latent_variance': latent_variance,
    }


def main():
    print("="*60)
    print("TEACHER MODEL EVALUATION")
    print("="*60)

    # Load model
    print("\nLoading model checkpoint...")
    checkpoint_path = 'checkpoints/last.ckpt'  # Within ear_teacher directory

    try:
        model = EarVAELightning.load_from_checkpoint(checkpoint_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"[OK] Model loaded successfully on {device}")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return

    print(f"\nModel Configuration:")
    print(f"  Latent dim: {model.hparams.latent_dim}")
    print(f"  KL weight: {model.hparams.kl_weight}")
    print(f"  Perceptual weight: {model.hparams.perceptual_weight}")
    print(f"  Contrastive weight: {model.hparams.contrastive_weight}")

    # Load validation data
    print("\nLoading validation data...")
    try:
        val_data = load_validation_data()
        print(f"[OK] Loaded {val_data.shape[0]} validation images")

        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    except Exception as e:
        print(f"[FAIL] Failed to load validation data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run evaluations
    results = {}

    # 1. Reconstruction quality
    recon_results = evaluate_reconstruction_quality(model, val_loader, device)
    results.update(recon_results)

    # 2. Feature discrimination
    discrim_results = evaluate_feature_discrimination(model, val_loader, device)
    results.update(discrim_results)

    # 3. Inference stability
    stability_results = evaluate_inference_stability(model, val_loader, device)
    results.update(stability_results)

    # Final summary
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)

    print(f"\n1. NaN/Inf Issues:")
    if results['nan_ratio'] == 0:
        print(f"   [OK] No NaN/Inf detected - model is stable")
    elif results['nan_ratio'] < 0.1:
        print(f"   [WARN] Minor NaN issues ({results['nan_ratio']*100:.1f}% batches)")
    else:
        print(f"   [FAIL] Significant NaN issues ({results['nan_ratio']*100:.1f}% batches)")

    print(f"\n2. Reconstruction Quality:")
    if not np.isnan(results['psnr_mean']):
        if results['psnr_mean'] >= 30:
            print(f"   [OK] EXCELLENT (PSNR: {results['psnr_mean']:.2f} dB)")
        elif results['psnr_mean'] >= 25:
            print(f"   [OK] GOOD (PSNR: {results['psnr_mean']:.2f} dB)")
        elif results['psnr_mean'] >= 20:
            print(f"   [WARN] ACCEPTABLE (PSNR: {results['psnr_mean']:.2f} dB)")
        else:
            print(f"   [FAIL] POOR (PSNR: {results['psnr_mean']:.2f} dB)")

        print(f"   KL Loss: {results['kl_mean']:.2f} (target: 100-300)")
        print(f"   SSIM: {results['ssim_mean']:.3f} (target: >0.8)")

    print(f"\n3. Teaching Capability:")
    if results['discrimination_score'] >= 0.7:
        print(f"   [OK] EXCELLENT - Ready to teach detection models")
    elif results['discrimination_score'] >= 0.5:
        print(f"   [OK] GOOD - Should work for detection")
    elif results['discrimination_score'] >= 0.3:
        print(f"   [WARN] MODERATE - May need improvement")
    else:
        print(f"   [FAIL] POOR - Not recommended for teaching")

    print(f"   Discrimination score: {results['discrimination_score']:.3f}")

    print(f"\n4. Inference Stability:")
    if results['stability_score'] >= 0.9:
        print(f"   [OK] STABLE - No issues detected")
    else:
        print(f"   [WARN] Some instability detected")

    # Overall recommendation
    print(f"\n{'OVERALL RECOMMENDATION:':<30}")

    if (results['nan_ratio'] == 0 and
        results['discrimination_score'] >= 0.5 and
        results['stability_score'] >= 0.5):
        print("  [OK] Model is ready for detection/landmark training")
        print("  [OK] Features are discriminative and stable")
    elif results['nan_ratio'] > 0.1:
        print("  [FAIL] Fix NaN issues before using for downstream tasks")
    elif results['discrimination_score'] < 0.3:
        print("  [WARN] Consider retraining with lower KL weight")
    else:
        print("  [WARN] Model is usable but could be improved")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
