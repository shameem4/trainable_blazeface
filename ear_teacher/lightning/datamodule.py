"""
PyTorch Lightning DataModule for Ear Teacher dataset.
"""

import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_dataloaders


class EarTeacherDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Ear Teacher VAE training.

    Handles data loading, augmentation, and batching.
    """

    def __init__(
        self,
        train_npy: Union[str, Path] = 'data/preprocessed/train_teacher.npy',
        val_npy: Union[str, Path] = 'data/preprocessed/val_teacher.npy',
        batch_size: int = 32,
        image_size: int = 128,
        num_workers: int = 4,
        root_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize DataModule.

        Args:
            train_npy: Path to training NPY metadata file
            val_npy: Path to validation NPY metadata file
            batch_size: Batch size for training and validation
            image_size: Target image size (square)
            num_workers: Number of data loading workers
            root_dir: Optional root directory for relative image paths
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_npy = Path(train_npy)
        self.val_npy = Path(val_npy)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.root_dir = Path(root_dir) if root_dir else None

        self.train_loader = None
        self.val_loader = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training or validation."""
        if stage == 'fit' or stage is None:
            self.train_loader, self.val_loader = get_dataloaders(
                train_npy=self.train_npy,
                val_npy=self.val_npy,
                batch_size=self.batch_size,
                image_size=self.image_size,
                num_workers=self.num_workers,
                root_dir=self.root_dir
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return self.train_loader

    def val_dataloader(self):
        """Return validation dataloader."""
        return self.val_loader
