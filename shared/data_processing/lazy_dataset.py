"""
Lazy loading dataset for NPZ files.

This module provides memory-efficient dataset classes that load data on-demand
from NPZ files, rather than loading everything into memory at once.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import warnings


class LazyNPZDataset:
    """
    Lazy loading dataset for NPZ files.

    Data is loaded on-demand when accessed by index, rather than loading
    the entire dataset into memory at once.

    Features:
    - Memory-efficient: only loads requested samples
    - Supports detector, landmarker, and teacher data formats
    - Thread-safe for data loaders with num_workers > 0
    - Caching support for frequently accessed samples

    Example:
        >>> dataset = LazyNPZDataset('data/preprocessed/train_detector.npz')
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['image', 'bboxes', 'image_path'])
    """

    def __init__(self, npz_path: Union[str, Path], cache_size: int = 0):
        """
        Initialize lazy loading dataset.

        Args:
            npz_path: Path to NPZ file
            cache_size: Number of samples to cache (0 = no caching)
        """
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        # Load NPZ file in mmap mode (memory-mapped, doesn't load data immediately)
        self._npz = np.load(self.npz_path, allow_pickle=True, mmap_mode='r')

        # Determine dataset type and keys
        self.keys = list(self._npz.keys())
        self.has_bboxes = 'bboxes' in self.keys
        self.has_keypoints = 'keypoints' in self.keys
        self.has_images = 'images' in self.keys
        self.has_image_paths = 'image_paths' in self.keys

        # Determine length
        if self.has_images:
            self._length = len(self._npz['images'])
        elif self.has_image_paths:
            self._length = len(self._npz['image_paths'])
        else:
            raise ValueError("NPZ file must contain 'images' or 'image_paths'")

        # Cache setup
        self.cache_size = cache_size
        self._cache = {} if cache_size > 0 else None
        self._cache_order = [] if cache_size > 0 else None

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample by index (lazy loading).

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample data (image, bboxes/keypoints, path)
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._length}")

        # Check cache first
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Load sample on-demand
        sample = {}

        if self.has_images:
            sample['image'] = self._npz['images'][idx]

        if self.has_bboxes:
            sample['bboxes'] = self._npz['bboxes'][idx]

        if self.has_keypoints:
            sample['keypoints'] = self._npz['keypoints'][idx]

        if self.has_image_paths:
            sample['image_path'] = str(self._npz['image_paths'][idx])

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
        batch = {}
        if self.has_images:
            batch['images'] = np.array([s['image'] for s in samples])
        if self.has_bboxes:
            batch['bboxes'] = np.array([s['bboxes'] for s in samples], dtype=object)
        if self.has_keypoints:
            batch['keypoints'] = np.array([s['keypoints'] for s in samples])
        if self.has_image_paths:
            batch['image_paths'] = np.array([s['image_path'] for s in samples])

        return batch

    def close(self):
        """Close the NPZ file handle."""
        if hasattr(self._npz, 'close'):
            self._npz.close()

    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        data_type = 'detector' if self.has_bboxes else 'landmarker' if self.has_keypoints else 'teacher'
        return f"LazyNPZDataset(path={self.npz_path.name}, type={data_type}, length={self._length}, cache_size={self.cache_size})"


class CombinedLazyDataset:
    """
    Combine multiple lazy datasets into one.

    Useful for combining train/val splits or multiple data sources.

    Example:
        >>> train_ds = LazyNPZDataset('train_detector.npz')
        >>> val_ds = LazyNPZDataset('val_detector.npz')
        >>> combined = CombinedLazyDataset([train_ds, val_ds])
        >>> print(len(combined))  # sum of both datasets
    """

    def __init__(self, datasets: List[LazyNPZDataset]):
        """
        Initialize combined dataset.

        Args:
            datasets: List of LazyNPZDataset instances
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

    def close(self):
        """Close all underlying datasets."""
        for ds in self.datasets:
            ds.close()

    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"CombinedLazyDataset(num_datasets={len(self.datasets)}, total_length={self._total_length})"


def load_lazy_dataset(npz_path: Union[str, Path], cache_size: int = 0) -> LazyNPZDataset:
    """
    Convenience function to load a lazy dataset.

    Args:
        npz_path: Path to NPZ file
        cache_size: Number of samples to cache (0 = no caching)

    Returns:
        LazyNPZDataset instance
    """
    return LazyNPZDataset(npz_path, cache_size=cache_size)
