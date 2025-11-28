"""
Ear Teacher Module - Convolutional VAE for learning ear representations.
"""

from .model import EarVAE, vae_loss
from .dataset import EarTeacherDataset, get_dataloaders

__all__ = ['EarVAE', 'vae_loss', 'EarTeacherDataset', 'get_dataloaders']
