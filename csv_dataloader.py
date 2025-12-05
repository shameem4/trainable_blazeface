"""
CSV-based data loader for BlazeFace training.

Loads face detection data from CSV format (WIDER Face style) and provides
train/val split functionality. Integrates with the anchor-based target encoding
from dataloader.py.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from dataloader import (
    encode_boxes_to_anchors,
    flatten_anchor_targets,
    collate_detector_fn
)
from utils import augmentation


class CSVDetectorDataset(Dataset):
    """
    Dataset for detector training from CSV file.

    CSV format:
    - image_path: relative path to image
    - x1, y1, w, h: bounding box in pixel coordinates
    - width, height: original image dimensions

    Following vincent1bt/blazeface-tensorflow:
    - Encodes boxes to anchor targets (896 anchors)
    - Returns flattened targets: (896, 5) with [class, ymin, xmin, ymax, xmax]
    - Supports data augmentation
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        target_size: Tuple[int, int] = (128, 128),
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with annotations
            root_dir: Root directory for image paths
            target_size: Target image size (height, width)
            augment: Whether to apply data augmentation
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.augment = augment

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Verify required columns
        required_cols = ['image_path', 'x1', 'y1', 'w', 'h']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        grouped = self.df.groupby('image_path', sort=False)
        self.samples = []
        for image_path, group in grouped:
            boxes = group[['x1', 'y1', 'w', 'h']].values.astype(np.float32)
            self.samples.append({
                'image_path': image_path,
                'boxes': boxes
            })

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.total_boxes = sum(len(sample['boxes']) for sample in self.samples)

        print(f"Loaded {len(self.samples)} images ({self.total_boxes} boxes) from {csv_path}")

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image."""
        full_path = self.root_dir / image_path

        # Load image
        image = cv2.imread(str(full_path))
        if image is None:
            raise ValueError(f"Could not load image: {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def _augment_image(
        self,
        image: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.

        Args:
            image: RGB image (H, W, 3)
            bboxes: (N, 4) boxes in [ymin, xmin, ymax, xmax] normalized format

        Returns:
            Augmented image and boxes
        """
        if not self.augment:
            return image, bboxes

        # Random saturation (50% chance)
        if np.random.random() > 0.5:
            image = augmentation.augment_saturation(image)

        # Random brightness (50% chance)
        if np.random.random() > 0.5:
            image = augmentation.augment_brightness(image)

        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5 and len(bboxes) > 0:
            image, bboxes = augmentation.augment_horizontal_flip(image, bboxes)

        return image, bboxes

    def _resize_and_pad(
        self,
        image: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize/pad image to target size while preserving aspect ratio."""
        target_h, target_w = self.target_size
        orig_h, orig_w = image.shape[:2]

        # Determine scaling (longer side fits target)
        if orig_h >= orig_w:
            scale = target_h / orig_h
            new_h = target_h
            new_w = int(round(orig_w * scale))
        else:
            scale = target_w / orig_w
            new_w = target_w
            new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h))

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        # Adjust boxes: convert to absolute coords, scale, pad, normalize
        if len(bboxes) > 0:
            abs_boxes = bboxes.copy()
            abs_boxes[:, [0, 2]] *= orig_h  # ymin, ymax
            abs_boxes[:, [1, 3]] *= orig_w  # xmin, xmax

            abs_boxes *= scale
            abs_boxes[:, [0, 2]] += pad_top
            abs_boxes[:, [1, 3]] += pad_left

            abs_boxes[:, [0, 2]] = np.clip(abs_boxes[:, [0, 2]], 0, target_h - 1)
            abs_boxes[:, [1, 3]] = np.clip(abs_boxes[:, [1, 3]], 0, target_w - 1)

            abs_boxes[:, [0, 2]] /= target_h
            abs_boxes[:, [1, 3]] /= target_w

            bboxes = abs_boxes

        return padded, bboxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dict with:
                - image: [3, H, W] tensor normalized to [0, 1]
                - anchor_targets: [896, 5] tensor [class, ymin, xmin, ymax, xmax]
                - small_anchors: [16, 16, 5] for visualization/debugging
                - big_anchors: [8, 8, 5] for visualization/debugging
        """
        sample_info = self.samples[idx]

        # Load image
        image = self._load_image(sample_info['image_path'])

        boxes_px = sample_info['boxes']
        orig_height, orig_width = image.shape[0], image.shape[1]

        if len(boxes_px) > 0:
            x1 = boxes_px[:, 0]
            y1 = boxes_px[:, 1]
            w = boxes_px[:, 2]
            h = boxes_px[:, 3]

            ymin = y1 / orig_height
            xmin = x1 / orig_width
            ymax = (y1 + h) / orig_height
            xmax = (x1 + w) / orig_width

            ymin = np.clip(ymin, 0, 1)
            xmin = np.clip(xmin, 0, 1)
            ymax = np.clip(ymax, 0, 1)
            xmax = np.clip(xmax, 0, 1)

            bboxes = np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        # Resize + pad to maintain aspect ratio (MediaPipe style)
        image, bboxes = self._resize_and_pad(image, bboxes)

        # Apply augmentation
        image, bboxes = self._augment_image(image, bboxes)

        # Encode boxes to anchor targets
        small_anchors, big_anchors = encode_boxes_to_anchors(
            bboxes, input_size=self.target_size[0]
        )

        # Flatten to (896, 5)
        anchor_targets = flatten_anchor_targets(small_anchors, big_anchors)

        # Convert to tensor (model handles normalization internally)
        # Ensure contiguous memory in case augmentations introduce negative strides
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            'image': image,
            'anchor_targets': torch.from_numpy(anchor_targets).float(),
            'small_anchors': torch.from_numpy(small_anchors).float(),
            'big_anchors': torch.from_numpy(big_anchors).float()
        }

    def get_sample_annotations(self, idx: int) -> Tuple[str, np.ndarray]:
        """Return original image path and GT boxes (x1, y1, w, h) for a sample."""
        sample = self.samples[idx]
        return sample['image_path'], sample['boxes'].copy()


def create_train_val_split(
    csv_path: str,
    output_dir: str,
    val_split: float = 0.2,
    random_seed: int = 42
):
    """
    Split CSV dataset into train and validation sets.

    Args:
        csv_path: Path to original CSV file
        output_dir: Directory to save train/val CSV files
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility
    """
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        raise ValueError("CSV must contain an 'image_path' column for splitting.")

    # Shuffle unique images so all rows from an image stay in the same split
    unique_images = df['image_path'].unique().tolist()
    if not unique_images:
        raise ValueError("No images found in CSV.")

    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_images)

    val_split = min(max(val_split, 0.0), 1.0)
    n_val = int(round(len(unique_images) * val_split))
    if len(unique_images) > 1:
        if n_val == 0:
            n_val = 1
        elif n_val >= len(unique_images):
            n_val = len(unique_images) - 1
    val_ids = set(unique_images[:n_val])

    val_df = df[df['image_path'].isin(val_ids)].reset_index(drop=True)
    train_df = df[~df['image_path'].isin(val_ids)].reset_index(drop=True)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    total_images = len(unique_images)
    print("Created train/val split:")
    print(
        f"  Train: {len(train_df)} rows "
        f"({len(train_df['image_path'].unique())}/{total_images} images) -> {train_path}"
    )
    print(
        f"  Val:   {len(val_df)} rows "
        f"({len(val_df['image_path'].unique())}/{total_images} images) -> {val_path}"
    )

    return train_path, val_path


def get_csv_dataloader(
    csv_path: str,
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (128, 128),
    augment: bool = True,
    max_samples: Optional[int] = None
) -> DataLoader:
    """
    Create DataLoader from CSV file.

    Args:
        csv_path: Path to CSV file
        root_dir: Root directory for image paths
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        target_size: Target image size
        augment: Whether to apply augmentation
        max_samples: Optional limit on samples

    Returns:
        DataLoader instance
    """
    dataset = CSVDetectorDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        target_size=target_size,
        augment=augment,
        max_samples=max_samples
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_detector_fn,
        pin_memory=True
    )


if __name__ == '__main__':
    # Example: Create train/val split
    import argparse

    parser = argparse.ArgumentParser(description='Split CSV dataset')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV file')
    parser.add_argument('--output', type=str, default='data/splits',
                        help='Output directory for split files')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split fraction')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    create_train_val_split(
        csv_path=args.csv,
        output_dir=args.output,
        val_split=args.val_split,
        random_seed=args.seed
    )
