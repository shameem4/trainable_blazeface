"""
Data loading utilities for ear detection and landmark models.

Provides Dataset classes for:
- Detector training (images + bounding boxes)
- Landmarker training (images + keypoints)
- Teacher training (images + boxes + landmarks)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
import cv2


class BaseEarDataset(Dataset):
    """Base dataset class for ear detection/landmark models."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        """
        Args:
            npy_path: Path to NPY file with annotations
            root_dir: Optional root directory for image paths
            transform: Optional transform to apply to images
            target_size: Target image size (height, width)
        """
        self.npy_path = Path(npy_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        self.target_size = target_size
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load and parse NPY metadata file."""
        metadata = np.load(self.npy_path, allow_pickle=True).item()
        
        self.image_paths = metadata['image_paths']
        self.metadata = metadata
        
        # Determine data type
        self._parse_annotations(metadata)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse annotations from metadata. Override in subclasses."""
        raise NotImplementedError
    
    def _get_image_path(self, idx: int) -> Path:
        """Get absolute image path."""
        image_path = Path(self.image_paths[idx])
        
        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path
        
        return image_path
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load and preprocess image."""
        image_path = self._get_image_path(idx)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _resize_image(
        self,
        image: np.ndarray,
        annotations: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image and scale annotations.
        
        Args:
            image: Original image
            annotations: Dict with bboxes/keypoints
            
        Returns:
            Resized image and scaled annotations
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Resize image
        resized = cv2.resize(image, (target_w, target_h))
        
        # Scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Scale annotations
        scaled_annotations = {}
        
        if 'bboxes' in annotations:
            bboxes = np.array(annotations['bboxes'])
            if len(bboxes) > 0:
                # Scale x, y, w, h
                bboxes[:, 0] *= scale_x  # x
                bboxes[:, 1] *= scale_y  # y
                bboxes[:, 2] *= scale_x  # w
                bboxes[:, 3] *= scale_y  # h
            scaled_annotations['bboxes'] = bboxes
        
        if 'keypoints' in annotations:
            keypoints = np.array(annotations['keypoints'])
            if len(keypoints) > 0:
                # Scale x, y coordinates
                keypoints[:, 0::2] *= scale_x  # x coords
                keypoints[:, 1::2] *= scale_y  # y coords
            scaled_annotations['keypoints'] = keypoints
        
        return resized, scaled_annotations
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class DetectorDataset(BaseEarDataset):
    """Dataset for detector training (images + bounding boxes)."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
        num_anchors: int = 896,
        num_keypoints: int = 6
    ):
        """
        Args:
            npy_path: Path to NPY file
            root_dir: Root directory for images
            transform: Optional image transform
            target_size: Target image size
            num_anchors: Number of anchor boxes
            num_keypoints: Number of keypoints per detection
        """
        self.num_anchors = num_anchors
        self.num_keypoints = num_keypoints
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse detector annotations."""
        self.bboxes = metadata.get('bboxes', [])
        self.keypoints = metadata.get('keypoints', [])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor
                - bboxes: [N, 4] tensor (x, y, w, h)
                - keypoints: [N, K*2] tensor (optional)
        """
        # Load image
        image = self._load_image(idx)
        
        # Get annotations
        annotations = {
            'bboxes': np.array(self.bboxes[idx]) if self.bboxes else np.array([])
        }
        if self.keypoints:
            annotations['keypoints'] = np.array(self.keypoints[idx])
        
        # Resize image and annotations
        image, annotations = self._resize_image(image, annotations)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: normalize to [0, 1] and convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert annotations to tensors
        sample = {
            'image': image,
            'bboxes': torch.from_numpy(annotations['bboxes']).float()
        }
        
        if 'keypoints' in annotations:
            sample['keypoints'] = torch.from_numpy(annotations['keypoints']).float()
        
        return sample


class LandmarkerDataset(BaseEarDataset):
    """Dataset for landmark training (images + keypoints)."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
        num_keypoints: int = 55
    ):
        """
        Args:
            npy_path: Path to NPY file
            root_dir: Root directory for images
            transform: Optional image transform
            target_size: Target image size
            num_keypoints: Number of keypoints to predict
        """
        self.num_keypoints = num_keypoints
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse landmark annotations."""
        self.keypoints = metadata.get('keypoints', [])
        self.visibility = metadata.get('visibility', None)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor
                - keypoints: [K, 2] tensor
                - visibility: [K] tensor (optional)
        """
        # Load image
        image = self._load_image(idx)
        
        # Get annotations
        keypoints = np.array(self.keypoints[idx]).reshape(-1, 2)
        annotations = {'keypoints': keypoints}
        
        # Resize image and annotations
        image, annotations = self._resize_image(image, annotations)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Normalize keypoints to [0, 1]
        kpts = annotations['keypoints'].reshape(-1, 2)
        kpts[:, 0] /= self.target_size[1]  # x / width
        kpts[:, 1] /= self.target_size[0]  # y / height
        
        sample = {
            'image': image,
            'keypoints': torch.from_numpy(kpts).float()
        }
        
        if self.visibility is not None:
            sample['visibility'] = torch.from_numpy(
                np.array(self.visibility[idx])
            ).float()
        
        return sample


class TeacherDataset(BaseEarDataset):
    """Dataset for teacher model training (images + boxes + landmarks)."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse teacher annotations (both boxes and landmarks)."""
        self.bboxes = metadata.get('bboxes', [])
        self.keypoints = metadata.get('keypoints', [])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor
                - bboxes: [N, 4] tensor
                - keypoints: [N, K, 2] tensor
        """
        # Load image
        image = self._load_image(idx)
        
        # Get annotations
        annotations = {}
        if self.bboxes:
            annotations['bboxes'] = np.array(self.bboxes[idx])
        if self.keypoints:
            annotations['keypoints'] = np.array(self.keypoints[idx])
        
        # Resize image and annotations
        image, annotations = self._resize_image(image, annotations)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        sample = {'image': image}
        
        if 'bboxes' in annotations:
            sample['bboxes'] = torch.from_numpy(annotations['bboxes']).float()
        if 'keypoints' in annotations:
            sample['keypoints'] = torch.from_numpy(annotations['keypoints']).float()
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-sized annotations.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Batched dict with padded annotations
    """
    # Stack images
    images = torch.stack([sample['image'] for sample in batch])
    
    result = {'image': images}
    
    # Handle bboxes (variable length per image)
    if 'bboxes' in batch[0]:
        # Find max number of boxes
        max_boxes = max(len(sample['bboxes']) for sample in batch)
        if max_boxes > 0:
            # Pad bboxes
            padded_bboxes = []
            for sample in batch:
                bboxes = sample['bboxes']
                if len(bboxes) < max_boxes:
                    padding = torch.zeros(max_boxes - len(bboxes), bboxes.shape[-1])
                    bboxes = torch.cat([bboxes, padding], dim=0)
                padded_bboxes.append(bboxes)
            result['bboxes'] = torch.stack(padded_bboxes)
            
            # Create mask for valid boxes
            result['bbox_mask'] = torch.stack([
                torch.cat([
                    torch.ones(len(sample['bboxes'])),
                    torch.zeros(max_boxes - len(sample['bboxes']))
                ])
                for sample in batch
            ])
    
    # Handle keypoints
    if 'keypoints' in batch[0]:
        result['keypoints'] = torch.stack([sample['keypoints'] for sample in batch])
    
    if 'visibility' in batch[0]:
        result['visibility'] = torch.stack([sample['visibility'] for sample in batch])
    
    return result


def get_dataloader(
    dataset_type: str,
    npy_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create a DataLoader.
    
    Args:
        dataset_type: One of 'detector', 'landmarker', 'teacher'
        npy_path: Path to NPY file
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for Dataset class
        
    Returns:
        DataLoader instance
    """
    dataset_map = {
        'detector': DetectorDataset,
        'landmarker': LandmarkerDataset,
        'teacher': TeacherDataset
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_map.keys())}")
    
    dataset = dataset_map[dataset_type](npy_path, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
