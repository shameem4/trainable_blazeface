"""
Lightning Module for ear teacher self-supervised training.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchvision.utils import make_grid, save_image

from ear_teacher.losses import ArcFaceLoss, CosFaceLoss
from ear_teacher.model import create_ear_teacher_model


class EarTeacherLightningModule(pl.LightningModule):
    """
    Lightning module for self-supervised ear teacher training.

    Uses reconstruction loss + metric learning (ArcFace/CosFace) to learn
    discriminative and detailed ear embeddings.
    """

    def __init__(
        self,
        pretrained_path: str = "models/convnext_tiny_22k_224.pth",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        reconstruction_weight: float = 1.0,
        metric_weight: float = 0.1,
        num_collage_samples: int = 10,
        metric_loss: str = "arcface",
        num_pseudo_classes: int = 512,
        arcface_margin: float = 0.5,
        arcface_scale: float = 64.0,
    ):
        """
        Args:
            pretrained_path: Path to pretrained ConvNeXt weights
            embedding_dim: Dimension of ConvNeXt output
            projection_dim: Dimension of projection head
            learning_rate: Peak learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            reconstruction_weight: Weight for reconstruction loss
            metric_weight: Weight for metric learning loss (ArcFace/CosFace)
            num_collage_samples: Number of samples for validation collage
            metric_loss: Type of metric loss ('arcface', 'cosface', or 'none')
            num_pseudo_classes: Number of pseudo-classes for metric learning
            arcface_margin: Angular margin for ArcFace (default: 0.5)
            arcface_scale: Feature scale for ArcFace (default: 64.0)
        """
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = create_ear_teacher_model(
            pretrained_path=pretrained_path,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
        )

        # Metric learning loss
        self.metric_loss_fn: Optional[nn.Module] = None
        if metric_loss == "arcface":
            self.metric_loss_fn = ArcFaceLoss(
                embedding_dim=embedding_dim,
                num_classes=num_pseudo_classes,
                margin=arcface_margin,
                scale=arcface_scale,
            )
        elif metric_loss == "cosface":
            self.metric_loss_fn = CosFaceLoss(
                embedding_dim=embedding_dim,
                num_classes=num_pseudo_classes,
                margin=arcface_margin,  # Will use as cosface margin
                scale=arcface_scale,
            )

        # Metrics
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_metric_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_metric_loss = MeanMetric()

        # Store validation samples for collage
        self.val_samples: List[Dict[str, torch.Tensor]] = []

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

    def reconstruction_loss(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruction loss (MSE).

        Args:
            reconstruction: Reconstructed images [B, 3, H, W] in range [-1, 1]
            target: Target images [B, 3, H, W] (normalized)

        Returns:
            Loss value
        """
        # Target is normalized with ImageNet stats, need to convert to [-1, 1] range
        # ImageNet normalization: (img - mean) / std
        # To reverse: img = normalized * std + mean
        mean = torch.tensor([0.485, 0.456, 0.406], device=target.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=target.device).view(1, 3, 1, 1)
        target_denorm = target * std + mean  # [0, 1]
        target_scaled = target_denorm * 2.0 - 1.0  # [-1, 1]

        # MSE loss
        return F.mse_loss(reconstruction, target_scaled)

    def shared_step(
        self,
        batch: Dict[str, Any],
        return_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Shared step for training and validation.

        Uses reconstruction + metric learning (ArcFace/CosFace).
        Pseudo-labels are generated from batch indices.
        """
        images = batch['image']
        batch_size = images.shape[0]

        # Forward pass
        output = self(images)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(output['reconstruction'], images)

        # Metric learning loss (ArcFace/CosFace)
        metric_loss = torch.tensor(0.0, device=images.device)
        if self.metric_loss_fn is not None:
            # Generate pseudo-labels from batch indices
            # Map batch indices to pseudo-class IDs
            pseudo_labels = torch.arange(batch_size, device=images.device) % self.hparams.num_pseudo_classes
            metric_loss = self.metric_loss_fn(output['embeddings'], pseudo_labels)

        # Combined loss
        total_loss = (
            self.hparams.reconstruction_weight * recon_loss +
            self.hparams.metric_weight * metric_loss
        )

        result = {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'metric_loss': metric_loss,
        }

        if return_outputs:
            result['reconstruction'] = output['reconstruction']
            result['images'] = images

        return result

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        losses = self.shared_step(batch)

        # Update metrics
        self.train_loss.update(losses['loss'])
        self.train_recon_loss.update(losses['recon_loss'])
        self.train_metric_loss.update(losses['metric_loss'])

        # Log
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', losses['recon_loss'], on_step=True, on_epoch=True)
        self.log('train/metric_loss', losses['metric_loss'], on_step=True, on_epoch=True)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)

        return losses['loss']

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        losses = self.shared_step(batch, return_outputs=True)

        # Update metrics
        self.val_loss.update(losses['loss'])
        self.val_recon_loss.update(losses['recon_loss'])
        self.val_metric_loss.update(losses['metric_loss'])

        # Log (use .compute() only after update)
        self.log('val/loss', losses['loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', losses['recon_loss'], on_step=False, on_epoch=True)
        self.log('val/metric_loss', losses['metric_loss'], on_step=False, on_epoch=True)

        # Store samples for collage (only if we need more)
        if len(self.val_samples) < self.hparams.num_collage_samples:
            needed = self.hparams.num_collage_samples - len(self.val_samples)
            batch_size = losses['images'].shape[0]
            for i in range(min(needed, batch_size)):
                self.val_samples.append({
                    'image': losses['images'][i].cpu(),
                    'reconstruction': losses['reconstruction'][i].cpu(),
                })

        return losses['loss']

    def on_validation_epoch_end(self):
        """Create reconstruction collage and reset metrics."""
        # Create collage if we have samples
        if len(self.val_samples) > 0:
            self.create_reconstruction_collage()

        # Clear samples for next epoch
        self.val_samples = []

        # Reset metrics
        self.val_loss.reset()
        self.val_recon_loss.reset()
        self.val_metric_loss.reset()

    def create_reconstruction_collage(self):
        """Create and save reconstruction collage."""
        # Get log directory
        log_dir = Path(self.trainer.log_dir) if self.trainer.log_dir else Path("outputs")
        collage_dir = log_dir / "reconstruction_collages"
        collage_dir.mkdir(parents=True, exist_ok=True)

        # Prepare images
        originals = []
        reconstructions = []

        for sample in self.val_samples[:self.hparams.num_collage_samples]:
            # Original (denormalize from ImageNet normalization)
            img = sample['image']
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img * std + mean
            originals.append(img_denorm)

            # Reconstruction (convert from [-1, 1] to [0, 1])
            recon = sample['reconstruction']
            recon_scaled = (recon + 1.0) / 2.0
            reconstructions.append(recon_scaled)

        # Stack images
        originals = torch.stack(originals)
        reconstructions = torch.stack(reconstructions)

        # Create 2-row grid: originals on top, reconstructions on bottom
        all_images = list(originals) + list(reconstructions)

        grid = make_grid(
            all_images,
            nrow=self.hparams.num_collage_samples,
            normalize=False,
            padding=2,
            pad_value=1.0,
        )

        # Save
        epoch = self.current_epoch
        save_path = collage_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, save_path)

        print(f"Saved reconstruction collage to {save_path}")

        # Update progression GIF
        self._update_progression_gif(collage_dir)

    def _update_progression_gif(self, collage_dir: Path):
        """Update GIF showing reconstruction progression across epochs."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            import glob
            import re

            # Collect all collage images
            collage_files = sorted(glob.glob(str(collage_dir / "epoch_*.png")))
            if len(collage_files) < 1:
                return

            # Load images and add epoch text
            frames = []
            for f in collage_files:
                img = PILImage.open(f).convert('RGB')
                
                # Extract epoch number from filename
                match = re.search(r'epoch_(\d+)', f)
                epoch_num = int(match.group(1)) if match else 0
                
                # Add epoch text overlay
                draw = ImageDraw.Draw(img)
                text = f"Epoch {epoch_num}"
                
                # Try to use a larger font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                # Draw text with background for visibility
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Position at top-left with padding
                x, y = 10, 10
                padding = 5
                draw.rectangle(
                    [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
                    fill='black'
                )
                draw.text((x, y), text, fill='white', font=font)
                
                frames.append(img)

            # Save GIF
            gif_path = collage_dir / "reconstruction_progression.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=500,  # 500ms per frame
                loop=0,  # Loop forever
            )
            print(f"Updated progression GIF: {gif_path}")

        except Exception as e:
            print(f"Failed to update progression GIF: {e}")

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
            if self.hparams.warmup_epochs > 0 and current_epoch < self.hparams.warmup_epochs:
                # Linear warmup
                return (current_epoch + 1) / self.hparams.warmup_epochs
            else:
                # Cosine annealing
                if self.hparams.max_epochs > self.hparams.warmup_epochs:
                    progress = (current_epoch - self.hparams.warmup_epochs) / (
                        self.hparams.max_epochs - self.hparams.warmup_epochs
                    )
                else:
                    progress = 0.0
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
        self.train_recon_loss.reset()
        self.train_metric_loss.reset()
