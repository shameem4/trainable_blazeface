"""
Dataset module for ear teacher model.
Handles loading, bbox cropping with buffer, and augmentations.
"""
import os
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class EarDataset(Dataset):
    """Dataset for loading ear images with bbox cropping."""

    def __init__(
        self,
        metadata_path: str,
        root_dir: str = ".",
        bbox_buffer: float = 0.10,
        image_size: int = 224,
        transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            metadata_path: Path to .npy file containing image_paths and bboxes
            root_dir: Root directory for resolving image paths
            bbox_buffer: Percentage buffer to add around bbox (default 10%)
            image_size: Target image size after cropping and resizing
            transform: Albumentations transform pipeline
        """
        self.root_dir = root_dir
        self.bbox_buffer = bbox_buffer
        self.image_size = image_size

        # Load metadata
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.image_paths = metadata['image_paths']
        self.bboxes = metadata['bboxes']

        # Convert Windows paths to Unix-style if needed
        self.image_paths = [path.replace('\\', os.sep) for path in self.image_paths]

        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def _add_bbox_buffer(
        self, bbox: np.ndarray, img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """Add buffer to bbox and clamp to image boundaries."""
        x1, y1, x2, y2 = bbox

        # Calculate buffer
        width = x2 - x1
        height = y2 - y1
        buffer_w = width * self.bbox_buffer
        buffer_h = height * self.bbox_buffer

        # Add buffer
        x1 = max(0, int(x1 - buffer_w))
        y1 = max(0, int(y1 - buffer_h))
        x2 = min(img_width, int(x2 + buffer_w))
        y2 = min(img_height, int(y2 + buffer_h))

        return x1, y1, x2, y2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process an ear image."""
        # Load image
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_height, img_width = image.shape[:2]

        # Get buffered bbox
        bbox = self.bboxes[idx]
        x1, y1, x2, y2 = self._add_bbox_buffer(bbox, img_width, img_height)

        # Crop to bbox
        cropped_image = image[y1:y2, x1:x2]

        # Apply transforms
        transformed = self.transform(image=cropped_image)
        image_tensor = transformed['image']

        return {
            'image': image_tensor,
            'image_path': self.image_paths[idx],
            'original_bbox': torch.tensor(bbox, dtype=torch.float32),
            'cropped_bbox': torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
        }


def get_default_transform(image_size: int = 224, augment: bool = False) -> A.Compose:
    """
    Get default transform pipeline.

    Args:
        image_size: Target image size
        augment: Whether to include augmentations (start with False, ramp up later)
    """
    if augment:
        # Future: Add augmentations here when ready to ramp up
        transforms = [
            A.Resize(image_size, image_size),
            # Placeholder for future augmentations:
            # A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            # A.RandomBrightnessContrast(p=0.3),
            # A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]
    else:
        # No augmentations for now
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ]

    return A.Compose(transforms)
