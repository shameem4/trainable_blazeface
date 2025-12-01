"""
DataModule for BlazeEar detector.
Handles train/val dataset loading with proper collation.
"""
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ear_detector.dataset import EarDetectorDataset, collate_fn, get_default_transform


class EarDetectorDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for ear detection.
    
    Handles train/val dataset loading with augmentations.
    """
    
    def __init__(
        self,
        train_metadata: str = "data/preprocessed/train_detector.npy",
        val_metadata: str = "data/preprocessed/val_detector.npy",
        root_dir: str = ".",
        image_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_metadata = train_metadata
        self.val_metadata = val_metadata
        self.root_dir = root_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = EarDetectorDataset(
                metadata_path=self.train_metadata,
                root_dir=self.root_dir,
                image_size=self.image_size,
                augment=self.augment,
            )
            
            # Check if val metadata exists
            val_path = Path(self.val_metadata)
            if val_path.exists():
                self.val_dataset = EarDetectorDataset(
                    metadata_path=self.val_metadata,
                    root_dir=self.root_dir,
                    image_size=self.image_size,
                    augment=False,  # No augmentation for validation
                )
            else:
                # Use subset of train for validation
                print(f"Warning: {self.val_metadata} not found, using train data for validation")
                self.val_dataset = EarDetectorDataset(
                    metadata_path=self.train_metadata,
                    root_dir=self.root_dir,
                    image_size=self.image_size,
                    augment=False,
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
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
