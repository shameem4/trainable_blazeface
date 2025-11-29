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


class DINOHybridEncoder(nn.Module):
    """
    Hybrid encoder combining DINOv2 pretrained features with custom conv layers and spatial attention.

    Architecture:
    - DINOv2-small backbone (frozen or fine-tunable) for initial features
    - Custom conv layers with residual blocks for ear-specific learning
    - Spatial attention modules at multiple scales

    Input: (B, 3, H, W) where H, W should be divisible by 16 (DINOv2 requirement)
    Output: (B, latent_dim * 2) for mu and logvar
    """

    def __init__(self, latent_dim: int = 512, image_size: int = 128, freeze_dino: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.freeze_dino = freeze_dino

        # DINOv2-small backbone
        # Note: Requires transformers library: pip install transformers
        try:
            from transformers import Dinov2Model
            self.dino_backbone = Dinov2Model.from_pretrained('facebook/dinov2-small')

            # Partially freeze DINOv2: freeze early layers, fine-tune last 4 blocks
            if freeze_dino:
                # Freeze everything
                for param in self.dino_backbone.parameters():
                    param.requires_grad = False
                print("DINOv2 backbone fully frozen")
            else:
                # Freeze embeddings and first 8 blocks, fine-tune last 4 blocks
                # This allows ear-specific adaptation while keeping low-level features stable
                for param in self.dino_backbone.embeddings.parameters():
                    param.requires_grad = False

                # DINOv2-small has 12 encoder blocks
                for i in range(8):  # Freeze first 8 blocks
                    for param in self.dino_backbone.encoder.layer[i].parameters():
                        param.requires_grad = False

                print("DINOv2 partially frozen: first 8 blocks frozen, last 4 blocks trainable")

        except ImportError:
            raise ImportError(
                "transformers library required for DINOv2. "
                "Install with: pip install transformers"
            )

        # DINOv2-small outputs 384 channels
        # For 128x128 input, DINOv2 outputs patches in a grid of approximately 9x9
        # (128 / 14 ≈ 9, where 14 is the patch size)

        # Custom conv layers on top of DINOv2 features
        # Input: 384 channels from DINOv2 (spatial size depends on input, ~9x9 for 128x128)
        # We'll use adaptive pooling to ensure consistent output size

        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )

        # Spatial attention after conv1
        self.attention1 = SpatialAttention(kernel_size=7)

        # Adaptive pooling to ensure 4x4 output regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512)
        )

        # Spatial attention after conv2
        self.attention2 = SpatialAttention(kernel_size=3)

        # Flatten and project to latent space (2x2 grid)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input images (B, 3, 128, 128)
            return_features: If True, return multi-scale feature maps for detection/landmarks

        Returns:
            If return_features=False:
                mu: Mean of latent distribution (B, latent_dim)
                logvar: Log variance of latent distribution (B, latent_dim)
            If return_features=True:
                mu, logvar, feature_pyramid (dict with keys: 'dino', 'feat1', 'feat2')
        """
        B = x.shape[0]

        # Extract features with DINOv2
        # DINOv2 expects images normalized with ImageNet stats, but we'll use our normalization
        dino_out = self.dino_backbone(x, interpolate_pos_encoding=True)

        # Get patch embeddings (skip CLS token)
        # last_hidden_state shape: (B, num_patches + 1, 384)
        features = dino_out.last_hidden_state[:, 1:, :]  # Skip CLS token

        # Reshape to spatial format
        # Calculate grid size dynamically
        num_patches = features.shape[1]
        grid_h = grid_w = int(num_patches ** 0.5)

        features = features.permute(0, 2, 1)  # (B, 384, num_patches)
        features = features.reshape(B, 384, grid_h, grid_w)  # (B, 384, H, W)
        dino_features = features  # Save for feature pyramid

        # Apply custom conv layers with spatial attention
        feat1 = self.conv1(features)
        feat1 = self.attention1(feat1)  # Focus on ear region at this scale (B, 512, ~9x9)

        # Adaptive pooling to ensure 4x4 size
        x = self.adaptive_pool(feat1)

        feat2 = self.conv2(x)  # 4x4 -> 2x2
        feat2 = self.attention2(feat2)  # Refine ear features at final scale (B, 512, 2x2)

        # Project to latent space
        x = self.flatten(feat2)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        if return_features:
            # Return multi-scale feature pyramid for detection/landmarks
            feature_pyramid = {
                'dino': dino_features,  # (B, 384, ~9x9) - High resolution DINOv2 features
                'feat1': feat1,         # (B, 512, ~9x9) - Mid resolution with attention
                'feat2': feat2,         # (B, 512, 2x2)  - Low resolution refined features
            }
            return mu, logvar, feature_pyramid

        return mu, logvar

    def extract_features(self, x: torch.Tensor):
        """
        Extract multi-scale spatial features WITHOUT going through latent bottleneck.

        Use this for detection/landmark tasks where you need spatial information.

        Args:
            x: Input images (B, 3, 128, 128)

        Returns:
            feature_pyramid: Dict with multi-scale feature maps
                - 'dino': (B, 384, ~9x9) - DINOv2 features
                - 'feat1': (B, 512, ~9x9) - Custom conv + attention features
                - 'feat2': (B, 512, 2x2) - Refined high-level features
        """
        _, _, feature_pyramid = self.forward(x, return_features=True)
        return feature_pyramid

    def load_imagenet_weights(self):
        """
        DINOv2 already comes with pretrained weights.
        This method is for compatibility with the standard encoder interface.
        """
        print("DINOv2 already uses pretrained weights from facebook/dinov2-small")
        print("  ✓ Loaded DINOv2 pretrained weights")


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
        Initialize VAE with DINOv2 hybrid encoder.

        Args:
            latent_dim: Dimensionality of latent space
            image_size: Input/output image size (must be multiple of 32)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Use DINOv2 hybrid encoder (partially unfrozen for faster learning)
        self.encoder = DINOHybridEncoder(latent_dim, image_size, freeze_dino=False)
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


class EarDetector(nn.Module):
    """
    Ear detection and landmark localization model using pretrained VAE encoder.

    This model leverages the DINOv2 + spatial attention encoder trained via VAE
    for downstream detection and landmark tasks.

    Architecture:
    - Pretrained DINOv2 hybrid encoder (frozen or fine-tunable)
    - Multi-scale feature pyramid
    - Detection head for bounding boxes
    - Keypoint head for 17 ear landmarks

    Usage:
        # After training VAE
        vae = EarVAE(...)
        vae.load_state_dict(torch.load('vae_checkpoint.pth'))

        # Create detector with pretrained encoder
        detector = EarDetector(vae.encoder, num_landmarks=17, freeze_encoder=False)

        # Fine-tune on labeled data
        bboxes, keypoints = detector(images)
    """

    def __init__(
        self,
        pretrained_encoder: DINOHybridEncoder,
        num_landmarks: int = 17,
        freeze_encoder: bool = False
    ):
        """
        Initialize detector with pretrained VAE encoder.

        Args:
            pretrained_encoder: Trained DINOHybridEncoder from VAE
            num_landmarks: Number of ear landmarks to predict
            freeze_encoder: If True, freeze encoder weights during fine-tuning
        """
        super().__init__()
        self.num_landmarks = num_landmarks

        # Use pretrained encoder
        self.encoder = pretrained_encoder

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen for detection fine-tuning")

        # Detection head on feat1 (512 channels, ~9x9 resolution)
        # Output: [x, y, w, h, confidence]
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(128, 5)  # [x, y, w, h, conf]
        )

        # Keypoint head on feat1 (512 channels, ~9x9 resolution)
        # Output: [x1, y1, x2, y2, ..., x17, y17] + confidence per point
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(256, num_landmarks * 3)  # x, y, visibility for each landmark
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for detection and landmark prediction.

        Args:
            x: Input images (B, 3, 128, 128)

        Returns:
            bboxes: Predicted bounding boxes (B, 5) [x, y, w, h, conf]
            keypoints: Predicted keypoints (B, num_landmarks * 3) [x1, y1, v1, ...]
        """
        # Extract multi-scale features from pretrained encoder
        features = self.encoder.extract_features(x)

        # Use feat1 for both tasks (~9x9 resolution, 512 channels)
        # This resolution is good for spatial localization
        feat = features['feat1']

        # Predict bounding box
        bboxes = self.bbox_head(feat)

        # Predict keypoints
        keypoints = self.keypoint_head(feat)

        return bboxes, keypoints

    @staticmethod
    def from_vae_checkpoint(checkpoint_path: str, num_landmarks: int = 17, freeze_encoder: bool = False):
        """
        Convenience method to create detector from trained VAE checkpoint.

        Args:
            checkpoint_path: Path to VAE checkpoint (.ckpt file)
            num_landmarks: Number of landmarks to predict
            freeze_encoder: Whether to freeze encoder weights

        Returns:
            EarDetector instance with pretrained encoder

        Example:
            detector = EarDetector.from_vae_checkpoint(
                'checkpoints/vae_epoch_50.ckpt',
                num_landmarks=17,
                freeze_encoder=False
            )
        """
        import torch

        # Load VAE checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Extract encoder state dict
        encoder_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.encoder.'):
                # Remove 'model.' prefix
                new_key = key.replace('model.', '')
                encoder_state[new_key] = value

        # Create new encoder and load weights
        encoder = DINOHybridEncoder(latent_dim=512, image_size=128)
        encoder.load_state_dict(encoder_state)

        print(f"Loaded pretrained encoder from {checkpoint_path}")

        # Create detector
        return EarDetector(encoder, num_landmarks, freeze_encoder)
