"""
Convolutional VAE model for learning ear representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpatialAttention(nn.Module):
    """
    Spatial attention module that focuses on important regions (ear center).
    Uses both max and average pooling to generate attention map.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Generate attention map from max and avg pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate and generate attention weights
        concat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(concat))  # (B, 1, H, W)

        # Apply attention weights
        return x * attention


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Encoder(nn.Module):
    """
    Convolutional encoder for VAE.

    Architecture: Progressive downsampling with residual blocks.
    Input: (B, 3, H, W) where H, W are multiples of 32
    Output: (B, latent_dim * 2) for mu and logvar
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Calculate output size after 5 stride-2 conv layers
        # image_size / (2^5) = image_size / 32
        self.final_size = image_size // 32
        self.flattened_size = 512 * self.final_size * self.final_size

        # Input: (B, 3, 128, 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # -> (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # (B, 64, 64, 64) -> (B, 128, 32, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128)
        )

        # (B, 128, 32, 32) -> (B, 256, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # (B, 256, 16, 16) -> (B, 512, 8, 8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        # Spatial attention after conv4 (at 8x8 resolution, good for focusing on ear region)
        self.attention4 = SpatialAttention(kernel_size=7)

        # (B, 512, 8, 8) -> (B, 512, 4, 4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )

        # Spatial attention after conv5 (at 4x4 resolution, final feature refinement)
        self.attention5 = SpatialAttention(kernel_size=3)

        # Flatten
        self.flatten = nn.Flatten()

        # Latent space projection
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input images (B, 3, 128, 128)

        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.attention4(x)  # Apply spatial attention at 8x8 resolution
        x = self.conv5(x)
        x = self.attention5(x)  # Apply spatial attention at 4x4 resolution
        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """
    Convolutional decoder for VAE.

    Architecture: Progressive upsampling with residual blocks.
    Input: (B, latent_dim)
    Output: (B, 3, H, W) where H, W are multiples of 32
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Calculate initial size (after 5 upsampling layers)
        self.initial_size = image_size // 32
        self.fc_output_size = 512 * self.initial_size * self.initial_size

        # Project latent to feature map
        self.fc = nn.Linear(latent_dim, self.fc_output_size)

        # (B, 512, 4, 4) -> (B, 512, 8, 8)
        self.deconv1 = nn.Sequential(
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # (B, 512, 8, 8) -> (B, 256, 16, 16)
        self.deconv2 = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # (B, 256, 16, 16) -> (B, 128, 32, 32)
        self.deconv3 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # (B, 128, 32, 32) -> (B, 64, 64, 64)
        self.deconv4 = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # (B, 64, 64, 64) -> (B, 3, 128, 128)
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.

        Args:
            z: Latent vector (B, latent_dim)

        Returns:
            Reconstructed image (B, 3, H, W)
        """
        x = self.fc(z)
        x = x.view(-1, 512, self.initial_size, self.initial_size)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        return x


class EarVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for ear representation learning.

    Features:
    - Deep convolutional architecture with residual connections
    - KL divergence regularization
    - Perceptual loss support (via external loss function)
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 256):
        """
        Initialize VAE.

        Args:
            latent_dim: Dimensionality of latent space
            image_size: Input/output image size (must be multiple of 32)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder = Encoder(latent_dim, image_size)
        self.decoder = Decoder(latent_dim, image_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)

        Returns:
            Sampled latent vector (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input images (B, 3, 128, 128)

        Returns:
            recon: Reconstructed images (B, 3, 128, 128)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent representation (deterministic).

        Args:
            x: Input images (B, 3, 128, 128)

        Returns:
            Latent vectors (B, latent_dim)
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.

        Args:
            z: Latent vectors (B, latent_dim)

        Returns:
            Reconstructed images (B, 3, 128, 128)
        """
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample random images from latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated images (num_samples, 3, 128, 128)
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
    recon_loss_type: str = 'mse'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute VAE loss = Reconstruction Loss + KL Divergence.

    Args:
        recon: Reconstructed images (B, 3, H, W)
        x: Original images (B, 3, H, W)
        mu: Latent mean (B, latent_dim)
        logvar: Latent log variance (B, latent_dim)
        kl_weight: Weight for KL divergence term
        recon_loss_type: Type of reconstruction loss ('mse' or 'l1')

    Returns:
        total_loss: Total VAE loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss
    if recon_loss_type == 'mse':
        recon_loss = F.mse_loss(recon, x, reduction='mean')
    elif recon_loss_type == 'l1':
        recon_loss = F.l1_loss(recon, x, reduction='mean')
    else:
        raise ValueError(f"Unknown reconstruction loss type: {recon_loss_type}")

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
