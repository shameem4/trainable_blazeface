"""
Lightning DataModule for ear teacher model.
"""
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ear_teacher.dataset import EarDataset, get_default_transform


class EarDataModule(pl.LightningDataModule):
    """DataModule for managing train and validation ear datasets."""

    def __init__(
        self,
        train_metadata_path: str = "common/data/preprocessed/train_teacher.npy",
        val_metadata_path: str = "common/data/preprocessed/val_teacher.npy",
        root_dir: str = ".",
        bbox_buffer: float = 0.10,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = False,
    ):
        """
        Args:
            train_metadata_path: Path to training metadata .npy file
            val_metadata_path: Path to validation metadata .npy file
            root_dir: Root directory for resolving image paths
            bbox_buffer: Percentage buffer around bbox (default 10%)
            image_size: Target image size
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            augment: Whether to use augmentations (start False, ramp up later)
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_metadata_path = train_metadata_path
        self.val_metadata_path = val_metadata_path
        self.root_dir = root_dir
        self.bbox_buffer = bbox_buffer
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

        self.train_dataset: Optional[EarDataset] = None
        self.val_dataset: Optional[EarDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if stage == 'fit' or stage is None:
            # Training dataset with optional augmentation
            train_transform = get_default_transform(
                image_size=self.image_size,
                augment=self.augment
            )
            self.train_dataset = EarDataset(
                metadata_path=self.train_metadata_path,
                root_dir=self.root_dir,
                bbox_buffer=self.bbox_buffer,
                image_size=self.image_size,
                transform=train_transform,
            )

            # Validation dataset (no augmentation)
            val_transform = get_default_transform(
                image_size=self.image_size,
                augment=False
            )
            self.val_dataset = EarDataset(
                metadata_path=self.val_metadata_path,
                root_dir=self.root_dir,
                bbox_buffer=self.bbox_buffer,
                image_size=self.image_size,
                transform=val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
