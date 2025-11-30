"""
Create Eigen Ears visualization from trained teacher model.

Extracts principal components from the latent space and visualizes
the dominant modes of variation in ear appearance.

Usage:
    cd ear_teacher
    python eigenears/create_eigenears.py
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning.module import EarVAELightning
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as T


def load_validation_data(max_samples=500):
    """Load validation data from preprocessed files."""
    possible_paths = [
        '../data/preprocessed/val_teacher.npy',
        '../../data/preprocessed/val_teacher.npy',
        'data/preprocessed/val_teacher.npy',
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"Loading data from: {path}")
            data = np.load(path, allow_pickle=True)

            if data.dtype == object:
                data_dict = data.item()
                if 'image_paths' in data_dict:
                    # Load images from paths
                    transform = T.Compose([
                        T.Resize((128, 128)),
                        T.ToTensor(),
                    ])

                    images = []
                    paths = data_dict['image_paths'][:max_samples]
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
                return torch.from_numpy(data[:max_samples]).float()

    raise FileNotFoundError("Could not find validation data")


def extract_latent_codes(model, data_loader, device):
    """Extract latent codes from all validation images."""
    print("\nExtracting latent codes...")
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

    latent_codes = np.vstack(all_mu)
    images = torch.cat(all_images, dim=0)

    print(f"Extracted {latent_codes.shape[0]} latent codes of dimension {latent_codes.shape[1]}")
    return latent_codes, images


def compute_eigenears(latent_codes, n_components=16):
    """Compute principal components of latent space."""
    print(f"\nComputing PCA with {n_components} components...")

    # Center the data
    mean_code = np.mean(latent_codes, axis=0)
    centered_codes = latent_codes - mean_code

    # Compute PCA
    pca = PCA(n_components=n_components)
    pca.fit(centered_codes)

    print(f"Explained variance ratio:")
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cumulative*100:.2f}%)")

    return pca, mean_code


def visualize_eigenear_component(model, pca, mean_code, component_idx, std_range=3.0, steps=9, device='cpu'):
    """Visualize one eigenear component by varying it along +/- std deviations."""
    print(f"\nVisualizing PC{component_idx+1}...")

    # Get the principal component
    pc = pca.components_[component_idx]
    std = np.sqrt(pca.explained_variance_[component_idx])

    # Create variations: mean - 3*std to mean + 3*std
    alphas = np.linspace(-std_range, std_range, steps)
    variations = []

    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            # Reconstruct latent code: mean + alpha * std * pc
            latent = mean_code + alpha * std * pc
            latent_tensor = torch.from_numpy(latent).float().unsqueeze(0).to(device)

            # Decode
            recon = model.model.decoder(latent_tensor)
            variations.append(recon.cpu())

    # Stack all variations
    variations = torch.cat(variations, dim=0)  # (steps, 3, H, W)

    return variations, alphas


def save_eigenear_grid(variations, alphas, component_idx, explained_var, output_dir):
    """Save a grid visualization of one eigenear component."""
    n_samples = variations.shape[0]

    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2.5))

    for i, (img, alpha) in enumerate(zip(variations, alphas)):
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        axes[i].set_title(f'{alpha:.1f}σ', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Eigenear PC{component_idx+1} ({explained_var*100:.2f}% variance)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'eigenear_pc{component_idx+1:02d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def save_all_eigenears_summary(pca, output_dir):
    """Save a summary visualization of all principal components."""
    n_components = pca.n_components_

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(n_components, 16)):
        pc = pca.components_[i]
        var = pca.explained_variance_ratio_[i]

        # Visualize the component vector as a heatmap
        # Reshape to approximate square
        side = int(np.sqrt(len(pc)))
        if side * side < len(pc):
            side += 1

        pc_padded = np.zeros(side * side)
        pc_padded[:len(pc)] = pc
        pc_grid = pc_padded.reshape(side, side)

        axes[i].imshow(pc_grid, cmap='RdBu_r', vmin=-np.abs(pc).max(), vmax=np.abs(pc).max())
        axes[i].set_title(f'PC{i+1} ({var*100:.1f}%)', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Eigenear Principal Components (Latent Space)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'eigenears_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved summary: {output_path}")


def create_eigenears_video_frames(model, pca, mean_code, component_idx, device='cpu', n_frames=60):
    """Create frames for an animated eigenear component (optional, for later use)."""
    print(f"\nCreating animation frames for PC{component_idx+1}...")

    pc = pca.components_[component_idx]
    std = np.sqrt(pca.explained_variance_[component_idx])

    # Create smooth oscillation: -3std -> +3std -> -3std
    t = np.linspace(0, 2 * np.pi, n_frames)
    alphas = 3.0 * np.sin(t)

    frames = []
    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            latent = mean_code + alpha * std * pc
            latent_tensor = torch.from_numpy(latent).float().unsqueeze(0).to(device)
            recon = model.model.decoder(latent_tensor)
            frames.append(recon.cpu())

    return torch.cat(frames, dim=0)


def main():
    print("="*60)
    print("EIGENEARS CREATION")
    print("="*60)

    # Create output directory
    output_dir = Path('ear_teacher/eigenears')
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load model
    print("\nLoading model checkpoint...")
    checkpoint_path = 'checkpoints/last.ckpt'

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
    print(f"  Image size: {model.hparams.image_size}")

    # Load validation data
    print("\nLoading validation data...")
    try:
        val_data = load_validation_data(max_samples=500)
        print(f"[OK] Loaded {val_data.shape[0]} validation images")

        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    except Exception as e:
        print(f"[FAIL] Failed to load validation data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Extract latent codes
    latent_codes, images = extract_latent_codes(model, val_loader, device)

    # Compute eigenears (PCA)
    n_components = 16
    pca, mean_code = compute_eigenears(latent_codes, n_components=n_components)

    # Save summary visualization
    save_all_eigenears_summary(pca, output_dir)

    # Visualize each component
    print("\n" + "="*60)
    print("GENERATING EIGENEAR VISUALIZATIONS")
    print("="*60)

    for i in range(n_components):
        variations, alphas = visualize_eigenear_component(
            model, pca, mean_code, i,
            std_range=3.0, steps=9, device=device
        )
        save_eigenear_grid(
            variations, alphas, i,
            pca.explained_variance_ratio_[i],
            output_dir
        )

    # Save PCA model for later use
    import pickle
    pca_path = output_dir / 'pca_model.pkl'
    with open(pca_path, 'wb') as f:
        pickle.dump({
            'pca': pca,
            'mean_code': mean_code,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'n_components': n_components,
            'latent_dim': model.hparams.latent_dim,
        }, f)
    print(f"\nSaved PCA model: {pca_path}")

    # Final summary
    print("\n" + "="*60)
    print("EIGENEARS CREATION COMPLETE")
    print("="*60)
    print(f"\nGenerated files in {output_dir}/:")
    print(f"  - eigenears_summary.png (overview of all components)")
    print(f"  - eigenear_pc01.png to eigenear_pc{n_components:02d}.png (individual visualizations)")
    print(f"  - pca_model.pkl (PCA model for later use)")
    print(f"\nTotal variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print(f"Top 5 components explain: {pca.explained_variance_ratio_[:5].sum()*100:.2f}%")


if __name__ == '__main__':
    main()
