"""PyTorch Lightning module for Ear VAE training."""

import torch
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanMetric
import torchvision
import math

from model import EarVAE, compute_vae_loss


class EarVAELightning(pl.LightningModule):
    """Lightning module for training Ear VAE with cyclic KL annealing."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
        image_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        kld_weight: float = 0.00025,
        kld_anneal_strategy: str = 'cyclic',
        kld_anneal_cycles: int = 4,
        kld_anneal_ratio: float = 0.5,
        kld_anneal_start: float = 0.0,
        kld_anneal_end: float = 1.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            latent_dim: Latent space dimensionality
            base_channels: Base number of channels
            image_size: Input image size
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            kld_weight: Final weight for KL divergence
            kld_anneal_strategy: Annealing strategy ('linear', 'cyclic', 'monotonic')
            kld_anneal_cycles: Number of cycles for cyclic annealing
            kld_anneal_ratio: Ratio of increasing phase in each cycle (0.5 = half increase, half constant)
            kld_anneal_start: Starting weight for KLD annealing
            kld_anneal_end: Ending weight multiplier for KLD annealing
        """
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = EarVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            base_channels=base_channels,
            image_size=image_size
        )

        # Metrics - Training
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_kld = MeanMetric()

        # Metrics - Validation
        self.val_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_kld = MeanMetric()
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Track global step for cyclic annealing
        self.training_step_count = 0

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def get_current_kld_weight(self):
        """
        Get current KLD weight with various annealing strategies.

        Supports:
        - 'linear': Simple linear increase from start to end
        - 'cyclic': Cyclical annealing with multiple cycles
        - 'monotonic': Monotonic increase with cosine schedule
        """
        strategy = self.hparams.kld_anneal_strategy

        if strategy == 'cyclic':
            return self._cyclic_annealing()
        elif strategy == 'monotonic':
            return self._monotonic_annealing()
        else:  # linear or default
            return self._linear_annealing()

    def _linear_annealing(self):
        """Simple linear annealing from start to end."""
        total_steps = self.trainer.max_epochs
        current_step = self.current_epoch

        if current_step >= total_steps:
            progress = 1.0
        else:
            progress = current_step / total_steps

        current_weight = (
            self.hparams.kld_anneal_start +
            (self.hparams.kld_anneal_end - self.hparams.kld_anneal_start) * progress
        )
        return current_weight * self.hparams.kld_weight

    def _monotonic_annealing(self):
        """
        Monotonic annealing with cosine schedule.
        Smoothly increases from start to end using cosine curve.
        """
        total_steps = self.trainer.max_epochs
        current_step = self.current_epoch

        if current_step >= total_steps:
            progress = 1.0
        else:
            # Cosine schedule: starts slow, accelerates in middle, slows at end
            progress = (1 - math.cos(math.pi * current_step / total_steps)) / 2

        current_weight = (
            self.hparams.kld_anneal_start +
            (self.hparams.kld_anneal_end - self.hparams.kld_anneal_start) * progress
        )
        return current_weight * self.hparams.kld_weight

    def _cyclic_annealing(self):
        """
        Cyclic KL annealing as described in "Cyclical Annealing Schedule: A Simple Approach
        to Mitigating KL Vanishing" (Fu et al., 2019).

        The KLD weight follows a cyclic pattern:
        - Increases linearly from 0 to target weight over ratio% of cycle
        - Stays at target weight for (1-ratio)% of cycle
        - Repeats for multiple cycles

        Benefits:
        - Prevents posterior collapse
        - Allows model to explore latent space
        - Improves reconstruction quality
        - Better balance between reconstruction and regularization
        """
        total_epochs = self.trainer.max_epochs
        n_cycles = self.hparams.kld_anneal_cycles
        ratio = self.hparams.kld_anneal_ratio

        # Calculate cycle parameters
        cycle_length = total_epochs / n_cycles
        current_position = self.current_epoch % cycle_length

        # Within each cycle, increase for ratio% of the cycle, then hold constant
        increase_length = cycle_length * ratio

        if current_position < increase_length:
            # Increasing phase
            progress = current_position / increase_length
        else:
            # Constant phase at maximum
            progress = 1.0

        # Apply start and end scaling
        current_weight = (
            self.hparams.kld_anneal_start +
            (self.hparams.kld_anneal_end - self.hparams.kld_anneal_start) * progress
        )

        return current_weight * self.hparams.kld_weight

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch

        # Forward pass
        reconstruction, mu, logvar = self(images)

        # Compute loss with annealed KLD weight
        current_kld_weight = self.get_current_kld_weight()
        loss, recon_loss, kld = compute_vae_loss(
            reconstruction, images, mu, logvar,
            kld_weight=current_kld_weight
        )

        # Update metrics
        self.train_loss(loss)
        self.train_recon_loss(recon_loss)
        self.train_kld(kld)

        # Increment step counter for cyclic annealing
        self.training_step_count += 1

        # Log metrics
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', self.train_recon_loss, on_step=False, on_epoch=True)
        self.log('train/kld', self.train_kld, on_step=False, on_epoch=True)
        self.log('train/kld_weight', current_kld_weight, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch

        # Forward pass
        reconstruction, mu, logvar = self(images)

        # Compute loss (use full KLD weight for validation)
        loss, recon_loss, kld = compute_vae_loss(
            reconstruction, images, mu, logvar,
            kld_weight=self.hparams.kld_weight
        )

        # Update metrics
        self.val_loss(loss)
        self.val_recon_loss(recon_loss)
        self.val_kld(kld)
        self.val_ssim(reconstruction, images)
        self.val_psnr(reconstruction, images)

        # Log metrics
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', self.val_recon_loss, on_step=False, on_epoch=True)
        self.log('val/kld', self.val_kld, on_step=False, on_epoch=True)
        self.log('val/ssim', self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

        # Log sample reconstructions (only for first batch)
        if batch_idx == 0:
            self._log_images(images, reconstruction)

        return loss

    def _log_images(self, images, reconstructions, max_images: int = 8):
        """Log sample images and reconstructions to tensorboard."""
        num_images = min(max_images, images.size(0))

        # Create comparison grid
        comparison = torch.cat([
            images[:num_images],
            reconstructions[:num_images]
        ])

        grid = torchvision.utils.make_grid(
            comparison,
            nrow=num_images,
            normalize=True,
            scale_each=True
        )

        self.logger.experiment.add_image(
            'val/reconstructions',
            grid,
            self.current_epoch
        )

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', current_lr, on_epoch=True)

    def encode(self, x):
        """Encode images to latent space (for inference)."""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(x)
            return mu

    def decode(self, z):
        """Decode latent vectors to images (for inference)."""
        self.eval()
        with torch.no_grad():
            return self.model.decode(z)

    def reconstruct(self, x):
        """Reconstruct images (for inference)."""
        self.eval()
        with torch.no_grad():
            reconstruction, _, _ = self.model(x)
            return reconstruction

    def sample(self, num_samples: int, device=None):
        """Sample random images from the learned distribution."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.hparams.latent_dim, device=device)
            samples = self.model.decode(z)
            return samples
