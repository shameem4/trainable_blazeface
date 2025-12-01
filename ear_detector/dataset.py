"""
Dataset module for BlazeEar detector.
Handles loading images and bounding box annotations with augmentations.
"""
import os
from typing import Dict, List, Optional, Tuple, Any

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class EarDetectorDataset(Dataset):
    """
    Dataset for ear detection training.
    
    Loads images and bounding boxes from metadata file.
    Applies augmentations and returns normalized images with target boxes.
    """
    
    def __init__(
        self,
        metadata_path: str,
        root_dir: str = ".",
        image_size: int = 128,
        transform: Optional[A.Compose] = None,
        augment: bool = False,
    ):
        """
        Args:
            metadata_path: Path to .npy file with image_paths and bboxes
            root_dir: Root directory for resolving image paths
            image_size: Target image size (128 for BlazeFace)
            transform: Optional custom albumentations transform
            augment: Whether to apply augmentations
        """
        self.root_dir = root_dir
        self.image_size = image_size
        
        # Load metadata
        metadata = np.load(metadata_path, allow_pickle=True).item()
        self.image_paths = metadata['image_paths']
        self.bboxes = metadata['bboxes']  # Shape: (N, num_boxes, 4) in x1,y1,w,h or x1,y1,x2,y2
        
        # Fix Windows paths
        self.image_paths = [p.replace('\\', os.sep) for p in self.image_paths]
        
        # Setup transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_default_transform(image_size, augment)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load image and annotations."""
        # Load image
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_height, img_width = image.shape[:2]
        
        # Get bboxes - handle (N, 1, 4) or (N, 4) shape
        bboxes = self.bboxes[idx]
        if bboxes.ndim == 2:
            bboxes = bboxes.reshape(-1, 4)
        else:
            bboxes = bboxes.reshape(-1, 4)
        
        # Convert bboxes from x,y,w,h to x1,y1,x2,y2 if needed
        # Check if w,h format (w and h would be smaller than x2,y2)
        converted_bboxes = []
        for bbox in bboxes:
            x1, y1, w_or_x2, h_or_y2 = bbox
            # Heuristic: if 3rd/4th values are much smaller than image size, it's w,h format
            if w_or_x2 < img_width * 0.8 and h_or_y2 < img_height * 0.8:
                # Assume x,y,w,h format
                x2 = x1 + w_or_x2
                y2 = y1 + h_or_y2
            else:
                # Already x1,y1,x2,y2
                x2 = w_or_x2
                y2 = h_or_y2
            converted_bboxes.append([x1, y1, x2, y2])
        
        # Normalize to [0, 1]
        normalized_bboxes = []
        for x1, y1, x2, y2 in converted_bboxes:
            normalized_bboxes.append([
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height,
            ])
        
        # Convert to albumentations format (x_min, y_min, x_max, y_max, class_label)
        labels = [0] * len(normalized_bboxes)  # All ears are class 0
        
        # Apply transforms
        transformed = self.transform(
            image=image,
            bboxes=normalized_bboxes,
            labels=labels,
        )
        
        image_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        
        # Convert bboxes to tensor
        if len(transformed_bboxes) > 0:
            boxes_tensor = torch.tensor(transformed_bboxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'boxes': boxes_tensor,
            'labels': torch.zeros(len(transformed_bboxes), dtype=torch.long),
            'image_path': self.image_paths[idx],
        }


def get_default_transform(image_size: int = 128, augment: bool = False) -> A.Compose:
    """
    Get default transform pipeline.
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentations
    """
    bbox_params = A.BboxParams(
        format='albumentations',  # [x_min, y_min, x_max, y_max] normalized
        label_fields=['labels'],
        min_visibility=0.3,
    )
    
    if augment:
        transforms = [
            A.Resize(image_size, image_size),
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            # Color augmentations
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.2),
            # Normalize to [-1, 1] (BlazeFace style)
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms, bbox_params=bbox_params)


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for variable number of boxes per image.
    """
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,
        'boxes': boxes,
        'labels': labels,
        'image_paths': image_paths,
    }
