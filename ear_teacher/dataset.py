"""Dataset module for ear teacher model.
Handles loading, bbox cropping with buffer, and augmentations.
"""
import os
import sys
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# Import bbox utilities
try:
    from shared.data_processing.bbox_utils import BBoxChecker, xywh_to_xyxy
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from shared.data_processing.bbox_utils import BBoxChecker, xywh_to_xyxy


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
        """Add buffer to bbox and clamp to image boundaries using BBoxChecker."""
        x1, y1, x2, y2 = bbox

        # Ensure bbox coordinates are valid (swap if inverted)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Calculate buffer
        width = x2 - x1
        height = y2 - y1
        buffer_w = width * self.bbox_buffer
        buffer_h = height * self.bbox_buffer

        # Add buffer
        x1_buffered = x1 - buffer_w
        y1_buffered = y1 - buffer_h
        x2_buffered = x2 + buffer_w
        y2_buffered = y2 + buffer_h

        # Use BBoxChecker to clamp to image bounds
        checker = BBoxChecker(min_width=50, min_height=50)
        clamped = checker.clamp_xyxy(
            [x1_buffered, y1_buffered, x2_buffered, y2_buffered],
            image_width=img_width,
            image_height=img_height
        )
        
        if clamped is not None:
            x1, y1, x2, y2 = [int(v) for v in clamped]
        else:
            # Fallback: use center crop with minimum dimensions
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            x1 = max(0, center_x - 25)
            y1 = max(0, center_y - 25)
            x2 = min(img_width, x1 + 50)
            y2 = min(img_height, y1 + 50)
            x1 = max(0, x2 - 50)
            y1 = max(0, y2 - 50)

        return x1, y1, x2, y2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process an ear image."""
        # Load image
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_height, img_width = image.shape[:2]

        # Get buffered bbox
        # bboxes[idx] is a list of bboxes for this image - teacher uses the first one
        bbox_list = self.bboxes[idx]
        bbox = np.array(bbox_list[0]) if isinstance(bbox_list, (list, np.ndarray)) and len(bbox_list) > 0 else bbox_list
        x1, y1, x2, y2 = self._add_bbox_buffer(bbox, img_width, img_height)

        # Crop to bbox
        cropped_image = image[y1:y2, x1:x2]

        # Apply transforms
        transformed = self.transform(image=cropped_image)
        image_tensor = transformed['image']

        return {
            'image': image_tensor,
            'image_path': self.image_paths[idx],
            'original_bbox': torch.tensor(bbox.astype(float), dtype=torch.float32),
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
        # Augmentations for training
        transforms = [
            A.Resize(image_size, image_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
            # Color augmentations
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Adaptive histogram equalization
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            # Noise/blur augmentations
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.2),
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
