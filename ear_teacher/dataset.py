"""
Dataset for Ear Teacher VAE with comprehensive augmentations.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from typing import Optional, Union


class EarTeacherDataset(Dataset):
    """
    Dataset for ear images with aggressive augmentations.

    Loads images from NPY metadata files (lazy loading) and applies:
    - Photometric jitter (brightness, contrast, saturation, hue)
    - Scale and translation jitter
    - Random rotations
    - Synthetic occlusions
    - Random cropping
    - Noise injection
    """

    def __init__(
        self,
        npy_path: Union[str, Path],
        image_size: int = 128,
        augment: bool = True,
        root_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize dataset.

        Args:
            npy_path: Path to NPY metadata file (train_teacher.npy or val_teacher.npy)
            image_size: Target image size (square)
            augment: Whether to apply augmentations
            root_dir: Optional root directory for relative image paths
        """
        self.npy_path = Path(npy_path)
        self.image_size = image_size
        self.augment = augment
        self.root_dir = Path(root_dir) if root_dir else None

        # Load metadata
        metadata = np.load(self.npy_path, allow_pickle=True).item()
        self.image_paths = metadata['image_paths']
        self.bboxes = metadata['bboxes']  # Single bbox per image for cropping

        # Build augmentation pipeline
        self.transform = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """Build albumentations transform pipeline."""
        if self.augment:
            return A.Compose([
                # Geometric augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1,      # Translation jitter ±10%
                    scale_limit=0.3,      # Scale jitter ±30%
                    rotate_limit=30,      # Rotation ±30°
                    border_mode=0,        # Constant border
                    p=0.8
                ),

                # Random crop and resize
                A.RandomResizedCrop(
                    height=self.image_size,
                    width=self.image_size,
                    scale=(0.7, 1.0),     # Crop 70-100% of image
                    ratio=(0.9, 1.1),     # Aspect ratio variation
                    p=0.6
                ),

                # Resize if RandomResizedCrop didn't apply
                A.Resize(self.image_size, self.image_size),

                # Photometric augmentations
                A.ColorJitter(
                    brightness=0.3,       # ±30% brightness
                    contrast=0.3,         # ±30% contrast
                    saturation=0.3,       # ±30% saturation
                    hue=0.1,              # ±10% hue
                    p=0.8
                ),

                # Advanced color transforms
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.ToGray(p=0.1),
                ], p=0.5),

                # Blur and noise
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                ], p=0.3),

                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

                # Synthetic occlusions
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        max_height=int(self.image_size * 0.2),
                        max_width=int(self.image_size * 0.2),
                        min_holes=3,
                        fill_value=0,
                        p=1.0
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        unit_size_min=8,
                        unit_size_max=16,
                        fill_value=0,
                        p=1.0
                    ),
                ], p=0.4),

                # Random flip
                A.HorizontalFlip(p=0.5),

                # Lighting effects
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),

                # Compression artifacts
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

                # Normalize and convert to tensor
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation: only resize and normalize
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and process image.

        Args:
            idx: Index

        Returns:
            Tensor of shape (3, H, W)
        """
        # Load image
        image_path = Path(self.image_paths[idx])
        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path

        image = Image.open(image_path).convert('RGB')

        # Crop using bbox
        bbox = self.bboxes[idx]
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Ensure valid crop
        img_w, img_h = image.size
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        # Crop to ear region
        image = image.crop((x, y, x + w, y + h))

        # Convert to numpy for albumentations
        image = np.array(image)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image']

        return image_tensor


def get_dataloaders(
    train_npy: Union[str, Path],
    val_npy: Union[str, Path],
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 4,
    root_dir: Optional[Union[str, Path]] = None
):
    """
    Create train and validation dataloaders.

    Args:
        train_npy: Path to training NPY file
        val_npy: Path to validation NPY file
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        root_dir: Optional root directory for image paths

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_dataset = EarTeacherDataset(
        train_npy,
        image_size=image_size,
        augment=True,
        root_dir=root_dir
    )

    val_dataset = EarTeacherDataset(
        val_npy,
        image_size=image_size,
        augment=False,
        root_dir=root_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
