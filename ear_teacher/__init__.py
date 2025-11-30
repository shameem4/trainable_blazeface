"""
Ear Teacher - Self-supervised learning for ear feature extraction.

This package provides a self-supervised learning framework for training
a teacher model on ear images. The learned representations will be
distilled to detector and landmarker models.
"""

from ear_teacher.datamodule import EarDataModule
from ear_teacher.dataset import EarDataset, get_default_transform
from ear_teacher.lightning_module import EarTeacherLightningModule
from ear_teacher.model import EarTeacherModel, create_ear_teacher_model

__version__ = "0.1.0"

__all__ = [
    "EarDataset",
    "get_default_transform",
    "EarDataModule",
    "EarTeacherModel",
    "create_ear_teacher_model",
    "EarTeacherLightningModule",
]
