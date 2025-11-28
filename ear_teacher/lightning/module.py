"""
PyTorch Lightning module for Ear VAE training.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.models import vgg16, VGG16_Weights
from typing import Optional, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import EarVAE, vae_loss


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss using VGG16 features.
    Captures high-level semantic similarity.
    """

    def __init__(self, layers: list = [3, 8, 15, 22]):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.layers = layers

        # Extract feature layers
        self.blocks = torch.nn.ModuleList()
        prev_layer = 0
        for layer_idx in layers:
            self.blocks.append(vgg[prev_layer:layer_idx + 1])
            prev_layer = layer_idx + 1

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y.

        Args:
            x: Predicted images (B, 3, H, W) in range [-1, 1]
            y: Target images (B, 3, H, W) in range [-1, 1]

        Returns:
            Perceptual loss value
        """
        # Denormalize from [-1, 1] to ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2

        x = (x - mean) / std
        y = (y - mean) / std

        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)

        return loss / len(self.blocks)


class EarVAELightning(pl.LightningModule):
    """
    PyTorch Lightning module for training Ear VAE.

    Features:
    - Multi-component loss (MSE + KL + Perceptual + SSIM)
    - Automatic metric tracking with torchmetrics
    - Learning rate scheduling with warmup
    - Reconstruction visualization
    """

    def __init__(
        self,
        latent_dim: int = 512,
        learning_rate: float = 1e-4,
        kl_weight: float = 0.0001,
        perceptual_weight: float = 0.5,
        ssim_weight: float = 0.1,
        recon_loss_type: str = 'mse',
        warmup_epochs: int = 5,
        scheduler: str = 'cosine'
    ):
        """
        Initialize Lightning module.

        Args:
            latent_dim: Latent space dimensionality
            learning_rate: Initial learning rate
            kl_weight: Weight for KL divergence loss
            perceptual_weight: Weight for perceptual loss
            ssim_weight: Weight for SSIM loss
            recon_loss_type: 'mse' or 'l1'
            warmup_epochs: Number of warmup epochs
            scheduler: LR scheduler type ('cosine', 'step', or 'none')
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = EarVAE(latent_dim=latent_dim)

        # Perceptual loss
        self.perceptual_loss = PerceptualLoss()

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)  # Range [-1, 1]
        self.psnr = PeakSignalNoiseRatio(data_range=2.0)

        # Validation metrics
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=2.0)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.model(x)

    def compute_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Args:
            recon: Reconstructed images
            x: Original images
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Dictionary of losses
        """
        # VAE loss (reconstruction + KL)
        _, recon_loss, kl_loss = vae_loss(
            recon, x, mu, logvar,
            kl_weight=self.hparams.kl_weight,
            recon_loss_type=self.hparams.recon_loss_type
        )

        # Perceptual loss
        perceptual = self.perceptual_loss(recon, x)

        # SSIM loss (1 - SSIM since we want to maximize SSIM)
        ssim_value = self.ssim(recon, x)
        ssim_loss = 1 - ssim_value

        # Total loss
        total_loss = (
            recon_loss +
            self.hparams.kl_weight * kl_loss +
            self.hparams.perceptual_weight * perceptual +
            self.hparams.ssim_weight * ssim_loss
        )

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'perceptual_loss': perceptual,
            'ssim_loss': ssim_loss,
            'ssim': ssim_value
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch
        recon, mu, logvar = self(x)

        losses = self.compute_loss(recon, x, mu, logvar)

        # Log metrics
        self.log('train/loss', losses['loss'], prog_bar=True)
        self.log('train/recon_loss', losses['recon_loss'])
        self.log('train/kl_loss', losses['kl_loss'])
        self.log('train/perceptual_loss', losses['perceptual_loss'])
        self.log('train/ssim', losses['ssim'], prog_bar=True)

        # Compute PSNR
        psnr = self.psnr(recon, x)
        self.log('train/psnr', psnr)

        return losses['loss']

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Validation step."""
        x = batch
        recon, mu, logvar = self(x)

        losses = self.compute_loss(recon, x, mu, logvar)

        # Log metrics
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', losses['recon_loss'], sync_dist=True)
        self.log('val/kl_loss', losses['kl_loss'], sync_dist=True)
        self.log('val/perceptual_loss', losses['perceptual_loss'], sync_dist=True)
        self.log('val/ssim', losses['ssim'], prog_bar=True, sync_dist=True)

        # Compute PSNR
        psnr = self.val_psnr(recon, x)
        self.log('val/psnr', psnr, sync_dist=True)

        # Log reconstructions (first batch only)
        if batch_idx == 0:
            self._log_reconstructions(x, recon)

    def _log_reconstructions(self, x: torch.Tensor, recon: torch.Tensor, num_images: int = 8):
        """Log reconstruction visualizations to tensorboard."""
        import torchvision

        # Take first num_images
        x = x[:num_images]
        recon = recon[:num_images]

        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        recon = (recon + 1) / 2

        # Create grid
        comparison = torch.cat([x, recon], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=False)

        # Log to tensorboard
        self.logger.experiment.add_image('reconstructions', grid, self.current_epoch)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )

        if self.hparams.scheduler == 'none':
            return optimizer

        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.learning_rate * 0.01
            )
        elif self.hparams.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        # Warmup scheduler
        if self.hparams.warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[self.hparams.warmup_epochs]
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        """Log learning rate at end of epoch."""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
