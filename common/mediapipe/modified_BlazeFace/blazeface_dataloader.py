# blazeface_dataloader.py
"""
Data loading for BlazeFace face detector.

Provides a dummy dataloader for testing/training and Lightning DataModule.
Replace the dummy data with real face detection datasets for actual training.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from .config import cfg_blazeface, MATCHING_CONFIG
from .blazeface_anchors import generate_anchors, compute_iou, anchors_to_xyxy


# =============================================================================
# Transforms
# =============================================================================

def get_default_transform(image_size: int = 128, augment: bool = False):
    """
    Get default transform pipeline.
    
    Args:
        image_size: Target image size
        augment: Whether to apply augmentations
        
    Returns:
        Albumentations Compose object or None if albumentations not installed
    """
    if not HAS_ALBUMENTATIONS:
        return None
    
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
                border_mode=0,  # cv2.BORDER_CONSTANT
                p=0.5,
            ),
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.5),
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
    
    return A.Compose(transforms, bbox_params=bbox_params, keypoint_params=A.KeypointParams(format='xy'))


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for variable number of faces per image.
    """
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    keypoints = [item['keypoints'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'image': images,
        'boxes': boxes,
        'keypoints': keypoints,
        'labels': labels,
    }


# =============================================================================
# Dummy Dataset
# =============================================================================

class DummyFaceDataset(Dataset):
    """
    Dummy dataset for testing BlazeFace training.
    
    Generates random images with synthetic face boxes and keypoints.
    Replace with real dataset for actual training.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Input image size (default 128)
        num_keypoints: Number of keypoints per face (default 6)
        max_faces: Maximum number of faces per image
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 128,
        num_keypoints: int = 6,
        max_faces: int = 3,
        transform: Optional[Any] = None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.max_faces = max_faces
        self.transform = transform
        
        # Pre-generate random data for consistency
        np.random.seed(42)
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Dict]:
        """Generate synthetic face data."""
        data = []
        
        for _ in range(self.num_samples):
            # Random number of faces (1-max_faces)
            num_faces = np.random.randint(1, self.max_faces + 1)
            
            # Generate random image (RGB)
            image = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
            
            boxes = []
            keypoints = []
            
            for _ in range(num_faces):
                # Random box (normalized coordinates)
                cx = np.random.uniform(0.2, 0.8)
                cy = np.random.uniform(0.2, 0.8)
                w = np.random.uniform(0.1, 0.4)
                h = np.random.uniform(0.1, 0.4)
                
                x1 = max(0, cx - w/2)
                y1 = max(0, cy - h/2)
                x2 = min(1, cx + w/2)
                y2 = min(1, cy + h/2)
                
                boxes.append([x1, y1, x2, y2])
                
                # Generate keypoints within the box
                # 6 keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
                kps = []
                for _ in range(self.num_keypoints):
                    kp_x = np.random.uniform(x1, x2)
                    kp_y = np.random.uniform(y1, y2)
                    kps.append([kp_x, kp_y])
                keypoints.append(kps)
            
            data.append({
                'image': image,
                'boxes': np.array(boxes, dtype=np.float32),
                'keypoints': np.array(keypoints, dtype=np.float32),
                'labels': np.ones(num_faces, dtype=np.int64),
            })
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        image = item['image'].copy()
        boxes = item['boxes'].copy()
        keypoints = item['keypoints'].copy()
        labels = item['labels'].copy()
        
        if self.transform is not None and HAS_ALBUMENTATIONS:
            # Flatten keypoints for albumentations
            kps_flat = keypoints.reshape(-1, 2).tolist()
            
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist(),
                keypoints=kps_flat,
                labels=labels.tolist(),
            )
            
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.long)
            
            # Reshape keypoints back
            kps_transformed = torch.tensor(transformed['keypoints'], dtype=torch.float32)
            num_faces = len(transformed['bboxes'])
            keypoints = kps_transformed.view(num_faces, self.num_keypoints, 2) if num_faces > 0 else torch.zeros(0, self.num_keypoints, 2)
        else:
            # Manual normalization without albumentations
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            keypoints = torch.tensor(keypoints, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Handle empty case
        if len(boxes) == 0:
            boxes = torch.zeros(0, 4, dtype=torch.float32)
            keypoints = torch.zeros(0, self.num_keypoints, 2, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'keypoints': keypoints,
            'labels': labels,
        }


# =============================================================================
# Lightning DataModule
# =============================================================================

class BlazeFaceDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for BlazeFace training.
    
    Uses dummy data by default. Override with real dataset paths for actual training.
    
    Args:
        train_samples: Number of training samples (dummy mode)
        val_samples: Number of validation samples (dummy mode)
        image_size: Input image size
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        augment: Whether to apply augmentations during training
    """
    
    def __init__(
        self,
        train_samples: int = 1000,
        val_samples: int = 200,
        image_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        train_transform = get_default_transform(self.image_size, augment=self.augment)
        val_transform = get_default_transform(self.image_size, augment=False)
        
        if stage == 'fit' or stage is None:
            self.train_dataset = DummyFaceDataset(
                num_samples=self.train_samples,
                image_size=self.image_size,
                transform=train_transform,
            )
            self.val_dataset = DummyFaceDataset(
                num_samples=self.val_samples,
                image_size=self.image_size,
                transform=val_transform,
            )
        
        if stage == 'validate':
            self.val_dataset = DummyFaceDataset(
                num_samples=self.val_samples,
                image_size=self.image_size,
                transform=val_transform,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    print("Testing BlazeFace DataModule...")
    
    # Create datamodule
    dm = BlazeFaceDataModule(
        train_samples=100,
        val_samples=20,
        batch_size=8,
        num_workers=0,
    )
    dm.setup()
    
    # Test train dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Boxes: {len(batch['boxes'])} items, first: {batch['boxes'][0].shape if len(batch['boxes'][0]) > 0 else 'empty'}")
    print(f"  Keypoints: {len(batch['keypoints'])} items, first: {batch['keypoints'][0].shape if len(batch['keypoints'][0]) > 0 else 'empty'}")
    print(f"  Labels: {len(batch['labels'])} items")
    
    print("\n✓ DataModule test passed!")
