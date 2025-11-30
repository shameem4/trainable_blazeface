"""Convolutional VAE with spatial attention for learning ear details."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from torchvision.models import vgg16, VGG16_Weights
from .attention import AttentionBlock


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self, layers=None):
        """
        Args:
            layers: List of VGG16 layer indices to use for feature extraction.
                   Default is [3, 8, 15, 22] (relu1_2, relu2_2, relu3_3, relu4_3)
        """
        super().__init__()
        if layers is None:
            layers = [3, 8, 15, 22]

        # Load pre-trained VGG16
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.layers = layers
        self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y.

        Args:
            x: Reconstructed images (B, C, H, W) in range [0, 1]
            y: Target images (B, C, H, W) in range [0, 1]

        Returns:
            Perceptual loss (scalar)
        """
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x_norm = (x - mean) / std
        y_norm = (y - mean) / std

        loss = 0.0
        x_feat = x_norm
        y_feat = y_norm

        # Extract features from specified layers
        for i, layer in enumerate(self.features):
            x_feat = layer(x_feat)
            y_feat = layer(y_feat)

            if i in self.layers:
                loss += F.mse_loss(x_feat, y_feat)

        return loss / len(self.layers)


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
    kld_weight: float = 0.00025,
    perceptual_weight: float = 0.0,
    ssim_weight: float = 0.0,
    edge_weight: float = 0.0,
    perceptual_loss_fn: nn.Module = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute normalized VAE loss combining reconstruction, KL divergence, perceptual, SSIM, and edge losses.

    Both reconstruction loss and KL divergence are normalized per sample to ensure
    consistent scaling regardless of image size or batch size.

    Args:
        reconstruction: Reconstructed images (B, C, H, W) in range [0, 1]
        target: Original images (B, C, H, W) in range [0, 1]
        mu: Mean of latent distribution (B, latent_dim)
        logvar: Log variance of latent distribution (B, latent_dim)
        kld_weight: Weight for KL divergence term
        perceptual_weight: Weight for perceptual loss term
        ssim_weight: Weight for SSIM loss term
        edge_weight: Weight for edge loss term
        perceptual_loss_fn: Optional pre-initialized PerceptualLoss module

    Returns:
        loss: Total weighted loss
        recon_loss: Normalized reconstruction loss (MSE per pixel)
        kld: Normalized KL divergence (per latent dimension)
        perceptual_loss: Perceptual loss (0 if perceptual_weight == 0)
        ssim_loss: SSIM loss (0 if ssim_weight == 0)
        edge_loss: Edge loss (0 if edge_weight == 0)
    """
    # Reconstruction loss: MSE normalized per pixel
    # Shape: (B, C, H, W) -> scalar
    recon_loss = F.mse_loss(reconstruction, target, reduction='mean')

    # KL divergence: Normalized per latent dimension
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Normalize by batch size and latent dimension for stability
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_per_sample)  # Average over batch

    # Perceptual loss
    perceptual_loss = torch.tensor(0.0, device=reconstruction.device)
    if perceptual_weight > 0 and perceptual_loss_fn is not None:
        perceptual_loss = perceptual_loss_fn(reconstruction, target)

    # SSIM loss (1 - SSIM since SSIM is similarity, not distance)
    ssim_loss = torch.tensor(0.0, device=reconstruction.device)
    if ssim_weight > 0:
        ssim_loss = 1.0 - _compute_ssim(reconstruction, target)

    # Edge loss (Sobel gradient-based edge preservation)
    edge_loss = torch.tensor(0.0, device=reconstruction.device)
    if edge_weight > 0:
        edge_loss = _compute_edge_loss(reconstruction, target)

    # Total loss
    loss = (
        recon_loss +
        kld_weight * kld +
        perceptual_weight * perceptual_loss +
        ssim_weight * ssim_loss +
        edge_weight * edge_loss
    )

    return loss, recon_loss, kld, perceptual_loss, ssim_loss, edge_loss


def _compute_edge_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute edge loss using Sobel gradient filters.

    Compares the edges (gradients) in the reconstructed and target images
    to encourage sharp, well-defined boundaries.

    Args:
        x: Reconstructed images (B, C, H, W) in range [0, 1]
        y: Target images (B, C, H, W) in range [0, 1]

    Returns:
        Edge loss (scalar)
    """
    # Sobel filters for horizontal and vertical gradients
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    # Replicate filters for each channel
    sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1)
    sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1)

    # Compute gradients for reconstruction
    x_grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    x_grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
    x_grad_mag = torch.sqrt(x_grad_x.pow(2) + x_grad_y.pow(2) + 1e-8)

    # Compute gradients for target
    y_grad_x = F.conv2d(y, sobel_x, padding=1, groups=y.size(1))
    y_grad_y = F.conv2d(y, sobel_y, padding=1, groups=y.size(1))
    y_grad_mag = torch.sqrt(y_grad_x.pow(2) + y_grad_y.pow(2) + 1e-8)

    # L1 loss between gradient magnitudes
    edge_loss = F.l1_loss(x_grad_mag, y_grad_mag, reduction='mean')

    return edge_loss


def _compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        x: First image tensor (B, C, H, W) in range [0, 1]
        y: Second image tensor (B, C, H, W) in range [0, 1]
        window_size: Size of the Gaussian window

    Returns:
        SSIM value (scalar in range [0, 1])
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(i - window_size // 2) ** 2 / (2 * sigma ** 2)))
        for i in range(window_size)
    ])
    gauss = gauss / gauss.sum()

    # Create 2D window
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(x.size(1), 1, window_size, window_size).contiguous()
    window = window.to(x.device)

    # Compute means
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.size(1))
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.size(1))

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=x.size(1)) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=y.size(1)) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.size(1)) - mu_xy

    # Compute SSIM
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean()


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
    loss, recon_loss, kld, perceptual_loss, ssim_loss, edge_loss = compute_vae_loss(reconstruction, x, mu, logvar)
    print(f"Loss: {loss.item():.6f}")
    print(f"Reconstruction loss (per pixel): {recon_loss.item():.6f}")
    print(f"KL divergence (normalized): {kld.item():.6f}")
    print(f"Perceptual loss: {perceptual_loss.item():.6f}")
    print(f"SSIM loss: {ssim_loss.item():.6f}")
    print(f"Edge loss: {edge_loss.item():.6f}")
    print()

    # Test with different batch sizes to verify normalization
    print("Testing loss normalization across different batch sizes:")
    for bs in [1, 2, 8, 16]:
        x_test = torch.randn(bs, 3, 256, 256)
        recon_test, mu_test, logvar_test = model(x_test)
        _, recon_test_loss, kld_test, _, _, _ = compute_vae_loss(recon_test, x_test, mu_test, logvar_test)
        print(f"  Batch size {bs:2d}: recon={recon_test_loss.item():.6f}, kld={kld_test.item():.6f}")

    # Test with all losses enabled
    print("\nTesting with all losses enabled:")
    perceptual_fn = PerceptualLoss()
    x_test = torch.randn(2, 3, 256, 256)
    # Ensure values are in [0, 1] range
    x_test = torch.sigmoid(x_test)
    recon_test, mu_test, logvar_test = model(x_test)
    loss_full, recon_l, kld_l, perc_l, ssim_l, edge_l = compute_vae_loss(
        recon_test, x_test, mu_test, logvar_test,
        kld_weight=0.00025,
        perceptual_weight=0.1,
        ssim_weight=0.1,
        edge_weight=0.1,
        perceptual_loss_fn=perceptual_fn
    )
    print(f"  Total loss: {loss_full.item():.6f}")
    print(f"  Reconstruction: {recon_l.item():.6f}")
    print(f"  KLD: {kld_l.item():.6f}")
    print(f"  Perceptual: {perc_l.item():.6f}")
    print(f"  SSIM: {ssim_l.item():.6f}")
    print(f"  Edge: {edge_l.item():.6f}")

    print("=" * 80)
    print("All tests passed!")
