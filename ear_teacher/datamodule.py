"""PyTorch Lightning DataModule for Ear dataset."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from .dataset import EarDataset, get_train_transform, get_val_transform


class EarDataModule(pl.LightningDataModule):
    """Lightning DataModule for Ear VAE training."""

    def __init__(
        self,
        train_data_path: str = "data/preprocessed/train_teacher.npy",
        val_data_path: str = "data/preprocessed/val_teacher.npy",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 256,
        pin_memory: bool = True,
    ):
        """
        Args:
            train_data_path: Path to training .npy file
            val_data_path: Path to validation .npy file
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            image_size: Image size for transforms
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """
        Download or prepare data (called only on main process).
        For this dataset, data should already be preprocessed.
        """
        # Check if data files exist
        train_path = Path(self.train_data_path)
        val_path = Path(self.val_data_path)

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                f"Please preprocess the data first."
            )

        if not val_path.exists():
            raise FileNotFoundError(
                f"Validation data not found at {val_path}. "
                f"Please preprocess the data first."
            )

    def setup(self, stage=None):
        """
        Setup datasets for training and validation.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Create transforms
        train_transform = get_train_transform(self.image_size)
        val_transform = get_val_transform(self.image_size)

        # Setup for training/validation
        if stage == 'fit' or stage is None:
            self.train_dataset = EarDataset(
                self.train_data_path,
                transform=train_transform,
                is_training=True
            )

            self.val_dataset = EarDataset(
                self.val_data_path,
                transform=val_transform,
                is_training=False
            )

        # Setup for validation only
        if stage == 'validate':
            self.val_dataset = EarDataset(
                self.val_data_path,
                transform=val_transform,
                is_training=False
            )

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True  # Drop last incomplete batch for stable training
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_num_samples(self):
        """Get number of training and validation samples."""
        if self.train_dataset is None or self.val_dataset is None:
            self.setup('fit')

        return {
            'train': len(self.train_dataset),
            'val': len(self.val_dataset)
        }
