"""Create a collage of input vs reconstructed images from validation dataset.

Run from project root:
    python ear_teacher/create_collage.py
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Ensure we're running from project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ear_teacher.lightning_module import EarVAELightning
from ear_teacher.dataset import EarDataset


def create_collage(checkpoint_path, val_data_path, output_path, num_samples=10, seed=42):
    """
    Create a collage of input vs reconstructed images.

    Args:
        checkpoint_path: Path to model checkpoint
        val_data_path: Path to validation data
        output_path: Path to save collage
        num_samples: Number of samples to show
        seed: Random seed for reproducibility
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}...")
    model = EarVAELightning.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Create transform that resizes to 256x256
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=1.0),
        ToTensorV2(),
    ])

    # Load validation dataset
    print(f"Loading validation data from {val_data_path}...")
    val_dataset = EarDataset(val_data_path, transform=transform, is_training=False)
    print(f"Validation dataset size: {len(val_dataset)}")

    # Fix paths to be absolute (if using on-the-fly loading)
    if val_dataset.image_paths is not None:
        print("Converting relative paths to absolute paths...")
        val_dataset.image_paths = [
            str(project_root / path) if not Path(path).is_absolute() else path
            for path in val_dataset.image_paths
        ]

    # Get images and reconstructions (with error handling for missing files)
    print("Generating reconstructions...")
    inputs = []
    reconstructions = []

    # Try to get num_samples valid images
    attempted = 0
    max_attempts = min(len(val_dataset), num_samples * 10)  # Try up to 10x to find valid images

    with torch.no_grad():
        while len(inputs) < num_samples and attempted < max_attempts:
            idx = np.random.randint(0, len(val_dataset))
            attempted += 1

            try:
                # Get input image
                img = val_dataset[idx]
                img_tensor = img.unsqueeze(0).to(device)

                # Get reconstruction
                recon, _, _ = model(img_tensor)

                # Move to CPU and convert to numpy
                inputs.append(img.cpu().numpy())
                reconstructions.append(recon.squeeze(0).cpu().numpy())

                if len(inputs) % 5 == 0:
                    print(f"  Processed {len(inputs)}/{num_samples} samples...")
            except Exception as e:
                # Debug: print first few errors
                if attempted <= 3:
                    print(f"    Error on idx {idx}: {type(e).__name__}: {str(e)[:100]}")
                continue

    if len(inputs) < num_samples:
        print(f"Warning: Could only generate {len(inputs)} samples out of {num_samples} requested")
        if len(inputs) == 0:
            print("Error: No valid images could be loaded. Check your data paths.")
            return

    num_samples = len(inputs)  # Update to actual number of samples obtained

    # Create collage
    print("Creating collage...")
    fig = plt.figure(figsize=(20, 4 * num_samples))
    gs = GridSpec(num_samples, 2, figure=fig, hspace=0.3, wspace=0.1)

    for i in range(num_samples):
        # Input image
        ax_input = fig.add_subplot(gs[i, 0])
        img_input = np.transpose(inputs[i], (1, 2, 0))
        # Denormalize from [-1, 1] to [0, 1]
        img_input = (img_input + 1) / 2
        img_input = np.clip(img_input, 0, 1)
        ax_input.imshow(img_input)
        ax_input.set_title(f'Input {i+1}', fontsize=14, fontweight='bold')
        ax_input.axis('off')

        # Reconstructed image
        ax_recon = fig.add_subplot(gs[i, 1])
        img_recon = np.transpose(reconstructions[i], (1, 2, 0))
        # Denormalize from [-1, 1] to [0, 1]
        img_recon = (img_recon + 1) / 2
        img_recon = np.clip(img_recon, 0, 1)
        ax_recon.imshow(img_recon)
        ax_recon.set_title(f'Reconstruction {i+1}', fontsize=14, fontweight='bold')
        ax_recon.axis('off')

    plt.suptitle('VAE Reconstructions: Input vs Output', fontsize=18, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Collage saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create input vs reconstruction collage')
    parser.add_argument('--checkpoint', type=str, default='ear_teacher/checkpoints/last.ckpt',
                       help='Path to model checkpoint (from project root)')
    parser.add_argument('--val_data', type=str, default='data/preprocessed/val_teacher.npy',
                       help='Path to validation data (from project root)')
    parser.add_argument('--output', type=str, default='ear_teacher/logs/reconstruction_collage.png',
                       help='Path to save output collage (from project root)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to show')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Resolve paths relative to project root
    checkpoint_path = project_root / args.checkpoint
    val_data_path = project_root / args.val_data
    output_path = project_root / args.output

    create_collage(
        checkpoint_path=str(checkpoint_path),
        val_data_path=str(val_data_path),
        output_path=str(output_path),
        num_samples=args.num_samples,
        seed=args.seed
    )
