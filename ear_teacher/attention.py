"""Spatial attention mechanisms for VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Spatial attention module that focuses on important spatial locations."""

    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor for attention computation
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attention-weighted features (B, C, H, W)
        """
        # Compute attention map
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))

        # Apply attention
        return x * attention


class ChannelSpatialAttention(nn.Module):
    """Combined channel and spatial attention mechanism."""

    def __init__(self, in_channels: int, reduction: int = 8):
        """
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction factor
        """
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )

        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attention-weighted features (B, C, H, W)
        """
        batch, channels, _, _ = x.size()

        # Channel attention
        avg_pool = self.avg_pool(x).view(batch, channels)
        max_pool = self.max_pool(x).view(batch, channels)

        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)

        channel_attention = torch.sigmoid(avg_out + max_out).view(batch, channels, 1, 1)
        x = x * channel_attention

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)

        spatial_attention = torch.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_attention

        return x


class AttentionBlock(nn.Module):
    """Convolutional block with spatial attention."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True,
                 use_attention: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            downsample: Whether to downsample (encoder) or upsample (decoder)
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 3,
                                           stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention = ChannelSpatialAttention(out_channels) if use_attention else None

    def forward(self, x):
        """Forward pass with optional attention."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.attention is not None:
            x = self.attention(x)

        return x
