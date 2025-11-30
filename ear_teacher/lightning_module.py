"""
Lightning Module for ear teacher self-supervised training.
"""
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric

from ear_teacher.model import create_ear_teacher_model


class EarTeacherLightningModule(pl.LightningModule):
    """
    Lightning module for self-supervised ear teacher training.

    Uses SimSiam-style self-supervised learning with augmented views.
    """

    def __init__(
        self,
        pretrained_path: str = "models/convnext_tiny_22k_224.pth",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        freeze_backbone: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
    ):
        """
        Args:
            pretrained_path: Path to pretrained ConvNeXt weights
            embedding_dim: Dimension of ConvNeXt output
            projection_dim: Dimension of projection head
            freeze_backbone: Whether to freeze backbone initially
            learning_rate: Peak learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
        """
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = create_ear_teacher_model(
            pretrained_path=pretrained_path,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            freeze_backbone=freeze_backbone,
        )

        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(x)

    def cosine_similarity_loss(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative cosine similarity loss (SimSiam style).

        Args:
            p: Predictions from one view [B, D]
            z: Projections from another view [B, D] (detached)

        Returns:
            Loss value
        """
        # Normalize
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        # Negative cosine similarity
        return -(p * z).sum(dim=1).mean()

    def shared_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Shared step for training and validation.

        For now, we use the same image twice (no augmentation yet).
        Later, when augmentations are added, we'll create two views.
        """
        images = batch['image']

        # Forward pass on same image (placeholder for future augmented views)
        # When augmentations are added, we'll create view1 and view2
        output1 = self(images)
        output2 = self(images)

        # SimSiam loss: predict one projection from another
        # D(p1, z2) + D(p2, z1) where D is negative cosine similarity
        loss1 = self.cosine_similarity_loss(
            output1['predictions'],
            output2['projections'].detach()
        )
        loss2 = self.cosine_similarity_loss(
            output2['predictions'],
            output1['projections'].detach()
        )

        # Symmetric loss
        loss = 0.5 * (loss1 + loss2)

        return loss

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        loss = self.shared_step(batch)

        # Update metrics
        self.train_loss(loss)

        # Log
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        loss = self.shared_step(batch)

        # Update metrics
        self.val_loss(loss)

        # Log
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return current_epoch / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                progress = (current_epoch - self.hparams.warmup_epochs) / (
                    self.hparams.max_epochs - self.hparams.warmup_epochs
                )
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def on_train_epoch_end(self):
        """Reset metrics at epoch end."""
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        """Reset metrics at epoch end."""
        self.val_loss.reset()
