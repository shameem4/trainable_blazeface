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
from model import EarVAE


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
    - Center-weighted loss to focus on ear region
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
        center_weight: float = 2.0,
        recon_loss_type: str = 'mse',
        warmup_epochs: int = 5,
        scheduler: str = 'cosine',
        image_size: int = 256
    ):
        """
        Initialize Lightning module.

        Args:
            latent_dim: Latent space dimensionality
            learning_rate: Initial learning rate
            kl_weight: Weight for KL divergence loss
            perceptual_weight: Weight for perceptual loss
            ssim_weight: Weight for SSIM loss
            center_weight: Weight multiplier for center region (higher = more focus on center)
            recon_loss_type: 'mse' or 'l1'
            warmup_epochs: Number of warmup epochs
            scheduler: LR scheduler type ('cosine', 'step', or 'none')
            image_size: Input image size
        """
        super().__init__()
        self.save_hyperparameters()

        # Model (uses DINOv2 hybrid encoder by default)
        self.model = EarVAE(latent_dim=latent_dim, image_size=image_size)

        # Perceptual loss
        self.perceptual_loss = PerceptualLoss()

        # Create center weight mask (gaussian-like, emphasizes center)
        self.register_buffer('center_mask', self._create_center_mask(image_size, center_weight))

        # Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)  # Range [-1, 1]
        self.psnr = PeakSignalNoiseRatio(data_range=2.0)

        # Validation metrics
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=2.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=2.0)

    def _create_center_mask(self, size: int, center_weight: float) -> torch.Tensor:
        """
        Create a 2D gaussian-like mask that emphasizes the center.

        Args:
            size: Image size (square)
            center_weight: Peak weight at center

        Returns:
            Tensor of shape (1, 1, size, size) with weights
        """
        # Create coordinate grids
        y = torch.linspace(-1, 1, size)
        x = torch.linspace(-1, 1, size)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # Compute distance from center
        dist = torch.sqrt(x_grid**2 + y_grid**2)

        # Create gaussian-like weight (sigma = 0.7 covers most of image)
        sigma = 0.7
        mask = torch.exp(-(dist**2) / (2 * sigma**2))

        # Scale: edges get weight 1.0, center gets center_weight
        mask = 1.0 + (center_weight - 1.0) * mask

        return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.model(x)

    def _create_occlusion_mask(self, x: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """
        Create mask for black regions (occlusions) in images.

        Args:
            x: Input images (B, 3, H, W) in range [-1, 1]
            threshold: Threshold for detecting black pixels (in [0, 1] range after denorm)

        Returns:
            Binary mask (B, 1, H, W) where 1 = valid pixel, 0 = occluded/black
        """
        # Denormalize from [-1, 1] to [0, 1]
        x_denorm = (x + 1) / 2

        # Check if all channels are below threshold (black pixels)
        # Use mean across channels to detect black regions
        grayscale = x_denorm.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        mask = (grayscale > threshold).float()

        return mask

    def compute_loss(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components with center-weighted reconstruction and occlusion masking.

        Args:
            recon: Reconstructed images
            x: Original images
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            Dictionary of losses
        """
        # Create occlusion mask (ignore black regions)
        occlusion_mask = self._create_occlusion_mask(x)  # (B, 1, H, W)

        # Center-weighted reconstruction loss
        if self.hparams.recon_loss_type == 'mse':
            pixel_loss = (recon - x) ** 2
        else:  # l1
            pixel_loss = torch.abs(recon - x)

        # Focal loss weighting: higher errors get more attention
        # This helps model focus on harder-to-reconstruct regions (ear details)
        focal_weight = (pixel_loss + 1e-8) ** 0.5  # Square root makes large errors more important

        # Apply focal weighting, then center weighting and occlusion masking
        # center_mask: (1, 1, H, W), occlusion_mask: (B, 1, H, W)
        combined_mask = self.center_mask * occlusion_mask
        weighted_loss = pixel_loss * focal_weight * combined_mask

        # Normalize by number of valid pixels to avoid bias
        num_valid_pixels = combined_mask.sum() + 1e-8
        recon_loss = weighted_loss.sum() / num_valid_pixels

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Perceptual loss (skip for now - VGG doesn't handle masked inputs well)
        # We could implement masked perceptual loss later if needed
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
        """Save reconstruction visualizations to disk."""
        import torchvision
        from pathlib import Path

        # Take first num_images
        x = x[:num_images]
        recon = recon[:num_images]

        # Denormalize from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        recon = (recon + 1) / 2

        # Create grid
        comparison = torch.cat([x, recon], dim=0)
        grid = torchvision.utils.make_grid(comparison, nrow=num_images, normalize=False)

        # Save to disk
        save_dir = Path(self.logger.log_dir) / 'reconstructions'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'epoch_{self.current_epoch:03d}.png'
        torchvision.utils.save_image(grid, save_path)

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
