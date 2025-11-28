"""
Lazy loading dataset for NPY metadata files.

This module provides memory-efficient dataset classes that load images on-demand
from disk based on paths stored in NPY files, rather than loading everything into memory.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Union, List


class LazyNPYDataset:
    """
    Lazy loading dataset for NPY metadata files.

    Images are loaded on-demand from disk when accessed by index, rather than
    storing all images in memory. The NPY file only contains metadata (paths + annotations).

    Features:
    - Memory-efficient: only loads requested images from disk
    - Supports detector, landmarker, and teacher data formats
    - Thread-safe for data loaders with num_workers > 0
    - Optional caching for frequently accessed samples
    - Automatic image loading and cropping (for teacher data)

    Example:
        >>> dataset = LazyNPYDataset('data/preprocessed/train_detector.npy')
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['image', 'bboxes', 'image_path'])
    """

    def __init__(self, npy_path: Union[str, Path], cache_size: int = 0,
                 image_loader='pil', root_dir: Optional[Union[str, Path]] = None):
        """
        Initialize lazy loading dataset.

        Args:
            npy_path: Path to NPY metadata file
            cache_size: Number of samples to cache (0 = no caching)
            image_loader: Image loading backend ('pil' or 'cv2')
            root_dir: Optional root directory for relative image paths
        """
        self.npy_path = Path(npy_path)
        if not self.npy_path.exists():
            raise FileNotFoundError(f"NPY file not found: {npy_path}")

        # Load metadata (lightweight - just paths and annotations)
        self._metadata = np.load(self.npy_path, allow_pickle=True).item()

        # Determine dataset type
        self.has_bboxes = 'bboxes' in self._metadata
        self.has_keypoints = 'keypoints' in self._metadata
        self.image_paths = self._metadata['image_paths']

        # Determine if teacher data (bboxes exist and are single bbox per image)
        self.is_teacher = False
        if self.has_bboxes:
            # Check if first bbox is single bbox (4 values) vs list of bboxes
            first_bbox = self._metadata['bboxes'][0]
            if isinstance(first_bbox, (list, np.ndarray)) and len(first_bbox) == 4:
                # Could be single bbox or list with one bbox
                if not isinstance(first_bbox[0], (list, np.ndarray)):
                    self.is_teacher = True

        self._length = len(self.image_paths)
        self.image_loader = image_loader
        self.root_dir = Path(root_dir) if root_dir else None

        # Cache setup
        self.cache_size = cache_size
        self._cache = {} if cache_size > 0 else None
        self._cache_order = [] if cache_size > 0 else None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self._length

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from disk."""
        # Resolve path
        image_path = Path(path)
        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.image_loader == 'cv2':
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # PIL
            img = Image.open(image_path).convert('RGB')
            img = np.array(img)

        return img

    def _crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop image using bounding box."""
        x, y, w, h = bbox
        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)
        x2 = min(img.shape[1], x + w)
        y2 = min(img.shape[0], y + h)

        return img[y:y2, x:x2]

    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample by index (lazy loading - loads image from disk).

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data (image loaded from disk, annotations, path)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._length}")

        # Check cache first
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Load image from disk
        image_path = str(self.image_paths[idx])
        image = self._load_image(image_path)

        # Build sample
        sample = {'image_path': image_path}

        if self.has_bboxes:
            bboxes = self._metadata['bboxes'][idx]
            if self.is_teacher:
                # Teacher data: crop image using bbox
                sample['bbox'] = bboxes
                sample['image'] = self._crop_image(image, bboxes)
            else:
                # Detector data: return full image + bboxes
                sample['bboxes'] = bboxes
                sample['image'] = image

        elif self.has_keypoints:
            # Landmarker data: full image + keypoints
            sample['keypoints'] = self._metadata['keypoints'][idx]
            sample['image'] = image

        # Update cache if enabled
        if self._cache is not None:
            self._update_cache(idx, sample)

        return sample

    def _update_cache(self, idx: int, sample: Dict):
        """Update LRU cache with new sample."""
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
        else:
            # Add new item
            if len(self._cache) >= self.cache_size:
                # Evict least recently used
                evict_idx = self._cache_order.pop(0)
                del self._cache[evict_idx]

            self._cache[idx] = sample
            self._cache_order.append(idx)

    def get_batch(self, indices: List[int]) -> Dict:
        """
        Get multiple samples as a batch.

        Args:
            indices: List of sample indices

        Returns:
            Dictionary with batched arrays
        """
        samples = [self[i] for i in indices]

        # Stack into batch
        batch = {
            'images': np.array([s['image'] for s in samples]),
            'image_paths': np.array([s['image_path'] for s in samples])
        }

        if self.has_bboxes:
            if self.is_teacher:
                batch['bboxes'] = np.array([s['bbox'] for s in samples], dtype=object)
            else:
                batch['bboxes'] = np.array([s['bboxes'] for s in samples], dtype=object)

        if self.has_keypoints:
            batch['keypoints'] = np.array([s['keypoints'] for s in samples], dtype=object)

        return batch

    def get_metadata(self, idx: int) -> Dict:
        """
        Get metadata only (no image loading).

        Args:
            idx: Sample index

        Returns:
            Dictionary with metadata (path, annotations)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._length}")

        sample = {'image_path': str(self.image_paths[idx])}

        if self.has_bboxes:
            sample['bboxes' if not self.is_teacher else 'bbox'] = self._metadata['bboxes'][idx]

        if self.has_keypoints:
            sample['keypoints'] = self._metadata['keypoints'][idx]

        return sample

    def __repr__(self) -> str:
        """String representation."""
        data_type = 'teacher' if self.is_teacher else 'detector' if self.has_bboxes else 'landmarker'
        return f"LazyNPYDataset(path={self.npy_path.name}, type={data_type}, length={self._length}, cache_size={self.cache_size})"


class CombinedLazyDataset:
    """
    Combine multiple lazy datasets into one.

    Useful for combining train/val splits or multiple data sources.

    Example:
        >>> train_ds = LazyNPYDataset('train_detector.npy')
        >>> val_ds = LazyNPYDataset('val_detector.npy')
        >>> combined = CombinedLazyDataset([train_ds, val_ds])
        >>> print(len(combined))  # sum of both datasets
    """

    def __init__(self, datasets: List[LazyNPYDataset]):
        """
        Initialize combined dataset.

        Args:
            datasets: List of LazyNPYDataset instances
        """
        if not datasets:
            raise ValueError("Must provide at least one dataset")

        self.datasets = datasets
        self._lengths = [len(ds) for ds in datasets]
        self._cumulative_lengths = np.cumsum([0] + self._lengths)
        self._total_length = sum(self._lengths)

    def __len__(self) -> int:
        """Return total number of samples."""
        return self._total_length

    def __getitem__(self, idx: int) -> Dict:
        """Get sample by global index."""
        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range for combined dataset of length {self._total_length}")

        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self._cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self._cumulative_lengths[dataset_idx]

        return self.datasets[dataset_idx][local_idx]

    def __repr__(self) -> str:
        """String representation."""
        return f"CombinedLazyDataset(num_datasets={len(self.datasets)}, total_length={self._total_length})"


def load_lazy_dataset(npy_path: Union[str, Path], cache_size: int = 0,
                     image_loader: str = 'pil', root_dir: Optional[Union[str, Path]] = None) -> LazyNPYDataset:
    """
    Convenience function to load a lazy dataset.

    Args:
        npy_path: Path to NPY file
        cache_size: Number of samples to cache (0 = no caching)
        image_loader: Image loading backend ('pil' or 'cv2')
        root_dir: Optional root directory for relative image paths

    Returns:
        LazyNPYDataset instance
    """
    return LazyNPYDataset(npy_path, cache_size=cache_size, image_loader=image_loader, root_dir=root_dir)
