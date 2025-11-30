"""PyTorch Lightning module for Ear VAE training."""

import torch
import pytorch_lightning as pl
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics import MeanMetric
import torchvision
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from .model import EarVAE, compute_vae_loss, PerceptualLoss


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
        perceptual_weight: float = 0.0,
        ssim_weight: float = 0.0,
        edge_weight: float = 0.0,
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
            perceptual_weight: Weight for perceptual loss (0.0 = disabled)
            ssim_weight: Weight for SSIM loss (0.0 = disabled)
            edge_weight: Weight for edge loss (0.0 = disabled)
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

        # Perceptual loss (only initialize if weight > 0)
        self.perceptual_loss = None
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()

        # Metrics - Training
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_kld = MeanMetric()
        self.train_perceptual_loss = MeanMetric()
        self.train_ssim_loss = MeanMetric()
        self.train_edge_loss = MeanMetric()
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Metrics - Validation
        self.val_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_kld = MeanMetric()
        self.val_perceptual_loss = MeanMetric()
        self.val_ssim_loss = MeanMetric()
        self.val_edge_loss = MeanMetric()
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)

        # Track global step for cyclic annealing
        self.training_step_count = 0

        # Store validation batch for reconstruction collage
        self.val_batch_for_collage = None

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
        loss, recon_loss, kld, perceptual_loss, ssim_loss, edge_loss = compute_vae_loss(
            reconstruction, images, mu, logvar,
            kld_weight=current_kld_weight,
            perceptual_weight=self.hparams.perceptual_weight,
            ssim_weight=self.hparams.ssim_weight,
            edge_weight=self.hparams.edge_weight,
            perceptual_loss_fn=self.perceptual_loss
        )

        # Check for NaN values
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(f"WARNING: NaN/Inf detected in training loss at step {batch_idx}")
            self.print(f"  recon_loss: {recon_loss.item()}, kld: {kld.item()}")
            self.print(f"  perceptual_loss: {perceptual_loss.item()}, ssim_loss: {ssim_loss.item()}")
            self.print(f"  edge_loss: {edge_loss.item()}")
            # Skip this batch by returning None (PyTorch Lightning will handle it)
            return None

        # Update metrics
        self.train_loss(loss)
        self.train_recon_loss(recon_loss)
        self.train_kld(kld)
        self.train_perceptual_loss(perceptual_loss)
        self.train_ssim_loss(ssim_loss)
        self.train_edge_loss(edge_loss)

        # Update image quality metrics (with NaN checking)
        if not (torch.isnan(reconstruction).any() or torch.isnan(images).any()):
            self.train_ssim(reconstruction, images)
            self.train_psnr(reconstruction, images)

        # Increment step counter for cyclic annealing
        self.training_step_count += 1

        # Log metrics
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', self.train_recon_loss, on_step=False, on_epoch=True)
        self.log('train/kld', self.train_kld, on_step=False, on_epoch=True)
        self.log('train/perceptual_loss', self.train_perceptual_loss, on_step=False, on_epoch=True)
        self.log('train/ssim_loss', self.train_ssim_loss, on_step=False, on_epoch=True)
        self.log('train/edge_loss', self.train_edge_loss, on_step=False, on_epoch=True)
        self.log('train/ssim', self.train_ssim, on_step=False, on_epoch=True)
        self.log('train/psnr', self.train_psnr, on_step=False, on_epoch=True)
        self.log('train/kld_weight', current_kld_weight, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch

        # Forward pass
        reconstruction, mu, logvar = self(images)

        # Compute loss (use full KLD weight for validation)
        loss, recon_loss, kld, perceptual_loss, ssim_loss, edge_loss = compute_vae_loss(
            reconstruction, images, mu, logvar,
            kld_weight=self.hparams.kld_weight,
            perceptual_weight=self.hparams.perceptual_weight,
            ssim_weight=self.hparams.ssim_weight,
            edge_weight=self.hparams.edge_weight,
            perceptual_loss_fn=self.perceptual_loss
        )

        # Check for NaN values
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(f"WARNING: NaN/Inf detected in validation loss at step {batch_idx}")
            self.print(f"  recon_loss: {recon_loss.item()}, kld: {kld.item()}")
            self.print(f"  perceptual_loss: {perceptual_loss.item()}, ssim_loss: {ssim_loss.item()}")
            self.print(f"  edge_loss: {edge_loss.item()}")
            # Skip this batch by returning None
            return None

        # Update metrics (only if not NaN)
        self.val_loss(loss)
        self.val_recon_loss(recon_loss)
        self.val_kld(kld)
        self.val_perceptual_loss(perceptual_loss)
        self.val_ssim_loss(ssim_loss)
        self.val_edge_loss(edge_loss)

        # Update image metrics with NaN checking
        if not (torch.isnan(reconstruction).any() or torch.isnan(images).any()):
            self.val_ssim(reconstruction, images)
            self.val_psnr(reconstruction, images)

        # Log metrics
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', self.val_recon_loss, on_step=False, on_epoch=True)
        self.log('val/kld', self.val_kld, on_step=False, on_epoch=True)
        self.log('val/perceptual_loss', self.val_perceptual_loss, on_step=False, on_epoch=True)
        self.log('val/ssim_loss', self.val_ssim_loss, on_step=False, on_epoch=True)
        self.log('val/edge_loss', self.val_edge_loss, on_step=False, on_epoch=True)
        self.log('val/ssim', self.val_ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/psnr', self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

        # Store first batch for reconstruction collage
        if batch_idx == 0 and self.val_batch_for_collage is None:
            self.val_batch_for_collage = (images.detach(), reconstruction.detach())

        # Log sample reconstructions (only for first batch)
        if batch_idx == 0:
            self._log_images(images, reconstruction)

        return loss

    def _log_images(self, images, reconstructions, max_images: int = 8):
        """Log sample images and reconstructions to logger (if supported)."""
        # Only log images if logger supports it (e.g., TensorBoardLogger)
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'add_image'):
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

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Save reconstruction collage
        if self.val_batch_for_collage is not None:
            self._save_reconstruction_collage()
            # Reset for next epoch
            self.val_batch_for_collage = None

    def _save_reconstruction_collage(self, num_samples: int = 10):
        """Save a collage of input vs reconstructed images."""
        if self.val_batch_for_collage is None:
            return

        images, reconstructions = self.val_batch_for_collage

        # Get logger directory (where metrics.csv is saved)
        if hasattr(self.logger, 'log_dir') and self.logger.log_dir is not None:
            log_dir = Path(self.logger.log_dir)
        else:
            # Fallback if logger doesn't have log_dir
            return

        # Create reconstructions directory
        recon_dir = log_dir / 'reconstructions'
        recon_dir.mkdir(parents=True, exist_ok=True)

        # Limit to num_samples
        num_samples = min(num_samples, images.size(0))
        images = images[:num_samples]
        reconstructions = reconstructions[:num_samples]

        # Move to CPU and convert to numpy
        images_np = images.cpu().numpy()
        reconstructions_np = reconstructions.cpu().numpy()

        # Create collage
        fig = plt.figure(figsize=(10, 2 * num_samples))
        gs = GridSpec(num_samples, 2, figure=fig, hspace=0.15, wspace=0.05)

        for i in range(num_samples):
            # Input image
            ax_input = fig.add_subplot(gs[i, 0])
            img_input = np.transpose(images_np[i], (1, 2, 0))
            # Denormalize from [-1, 1] to [0, 1]
            img_input = (img_input + 1) / 2
            img_input = np.clip(img_input, 0, 1)
            ax_input.imshow(img_input)
            if i == 0:
                ax_input.set_title('Input', fontsize=12, fontweight='bold')
            ax_input.axis('off')

            # Reconstructed image
            ax_recon = fig.add_subplot(gs[i, 1])
            img_recon = np.transpose(reconstructions_np[i], (1, 2, 0))
            # Denormalize from [-1, 1] to [0, 1]
            img_recon = (img_recon + 1) / 2
            img_recon = np.clip(img_recon, 0, 1)
            ax_recon.imshow(img_recon)
            if i == 0:
                ax_recon.set_title('Reconstruction', fontsize=12, fontweight='bold')
            ax_recon.axis('off')

        plt.suptitle(f'Epoch {self.current_epoch} - Validation Reconstructions',
                     fontsize=14, fontweight='bold', y=0.998)

        # Save figure
        output_path = recon_dir / f'epoch_{self.current_epoch:03d}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)

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
