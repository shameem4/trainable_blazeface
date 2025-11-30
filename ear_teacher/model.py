"""Convolutional VAE with spatial attention for learning ear details."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from attention import AttentionBlock


class EarVAE(nn.Module):
    """Variational Autoencoder with spatial attention for ear image modeling."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
        image_size: int = 256
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            latent_dim: Dimensionality of the latent space
            base_channels: Base number of channels (will be multiplied in deeper layers)
            image_size: Input image size (assumed square)
        """
        super().__init__()

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.encoded_spatial_size = image_size // (2 ** 4)  # 4 downsampling layers

        # Encoder: Progressive downsampling with attention
        self.encoder = self._build_encoder(in_channels, base_channels)
        self.encoder_se = SqueezeExcitation(base_channels * 8, base_channels * 8 // 4)

        # Latent space projection
        self.flatten_size = base_channels * 8 * self.encoded_spatial_size ** 2
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder projection
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder_se = SqueezeExcitation(base_channels * 8, base_channels * 8 // 4)

        # Decoder: Progressive upsampling with attention
        self.decoder = self._build_decoder(base_channels)

        # Final reconstruction layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def _build_encoder(self, in_channels: int, base_channels: int) -> nn.ModuleList:
        """Build encoder with progressive downsampling."""
        return nn.ModuleList([
            AttentionBlock(in_channels, base_channels, downsample=True, use_attention=False),
            AttentionBlock(base_channels, base_channels * 2, downsample=True, use_attention=True),
            AttentionBlock(base_channels * 2, base_channels * 4, downsample=True, use_attention=True),
            AttentionBlock(base_channels * 4, base_channels * 8, downsample=True, use_attention=True),
        ])

    def _build_decoder(self, base_channels: int) -> nn.ModuleList:
        """Build decoder with progressive upsampling."""
        return nn.ModuleList([
            AttentionBlock(base_channels * 8, base_channels * 4, downsample=False, use_attention=True),
            AttentionBlock(base_channels * 4, base_channels * 2, downsample=False, use_attention=True),
            AttentionBlock(base_channels * 2, base_channels, downsample=False, use_attention=True),
            AttentionBlock(base_channels, base_channels, downsample=False, use_attention=False),
        ])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            mu, logvar: Mean and log variance of latent distribution
        """
        for layer in self.encoder:
            x = layer(x)

        x = self.encoder_se(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Reconstructed image (B, C, H, W)
        """
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, self.encoded_spatial_size, self.encoded_spatial_size)

        x = self.decoder_se(x)

        for layer in self.decoder:
            x = layer(x)

        return self.final_layer(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


def compute_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kld_weight: float = 0.00025
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute normalized VAE loss combining reconstruction and KL divergence.

    Both reconstruction loss and KL divergence are normalized per sample to ensure
    consistent scaling regardless of image size or batch size.

    Args:
        reconstruction: Reconstructed images (B, C, H, W)
        target: Original images (B, C, H, W)
        mu: Mean of latent distribution (B, latent_dim)
        logvar: Log variance of latent distribution (B, latent_dim)
        kld_weight: Weight for KL divergence term

    Returns:
        loss: Total weighted loss
        recon_loss: Normalized reconstruction loss (MSE per pixel)
        kld: Normalized KL divergence (per latent dimension)
    """
    # Reconstruction loss: MSE normalized per pixel
    # Shape: (B, C, H, W) -> scalar
    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')

    # KL divergence: Normalized per latent dimension
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Normalize by batch size and latent dimension for stability
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_per_sample)  # Average over batch

    # Total loss
    loss = recon_loss + kld_weight * kld

    return loss, recon_loss, kld


# Alias for backward compatibility
vae_loss = compute_vae_loss


if __name__ == "__main__":
    # Test the model
    print("Testing EarVAE Model")
    print("=" * 80)

    model = EarVAE(in_channels=3, latent_dim=256, base_channels=64, image_size=256)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    reconstruction, mu, logvar = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print()

    # Test loss computation
    loss, recon_loss, kld = compute_vae_loss(reconstruction, x, mu, logvar)
    print(f"Loss: {loss.item():.6f}")
    print(f"Reconstruction loss (per pixel): {recon_loss.item():.6f}")
    print(f"KL divergence (normalized): {kld.item():.6f}")
    print()

    # Test with different batch sizes to verify normalization
    print("Testing loss normalization across different batch sizes:")
    for bs in [1, 2, 8, 16]:
        x_test = torch.randn(bs, 3, 256, 256)
        recon_test, mu_test, logvar_test = model(x_test)
        _, recon_test_loss, kld_test = compute_vae_loss(recon_test, x_test, mu_test, logvar_test)
        print(f"  Batch size {bs:2d}: recon={recon_test_loss.item():.6f}, kld={kld_test.item():.6f}")

    print("=" * 80)
    print("All tests passed!")
