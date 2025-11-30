"""Ear Teacher module for learning intricate ear details using VAE with spatial attention."""

from .model import EarVAE, compute_vae_loss
from .dataset import EarDataset, get_train_transform, get_val_transform
from .attention import AttentionBlock, ChannelSpatialAttention
from .lightning_module import EarVAELightning
from .datamodule import EarDataModule

__all__ = [
    'EarVAE',
    'compute_vae_loss',
    'EarDataset',
    'get_train_transform',
    'get_val_transform',
    'AttentionBlock',
    'ChannelSpatialAttention',
    'EarVAELightning',
    'EarDataModule',
]
