# __init__.py
"""
BlazeFace Face Detector Module.

Provides face detection with keypoint localization using the BlazeFace architecture.

Main Components:
- BlazeFace: Face detection model
- BlazeFaceLightningModule: Lightning wrapper for training
- FaceDetector: Inference wrapper
- BlazeFaceDataModule: Data loading

Quick Start:
    # Inference
    from common.mediapipe.modified_BlazeFace import FaceDetector
    detector = FaceDetector("checkpoint.ckpt")
    results = detector.detect("image.jpg")
    
    # Training
    from common.mediapipe.modified_BlazeFace import BlazeFaceLightningModule, BlazeFaceDataModule
    model = BlazeFaceLightningModule()
    datamodule = BlazeFaceDataModule()
"""

# Model architecture
from .blazeface import BlazeFace, BlazeBlock, BlazeFaceBackbone

# Configuration
from .config import cfg_blazeface, cfg_blazeface_front, MATCHING_CONFIG

# Anchors and utilities
from .blazeface_anchors import (
    generate_anchors,
    AnchorGenerator,
    encode_boxes,
    decode_boxes,
    decode_keypoints,
    compute_iou,
    match_anchors,
    anchors_to_xyxy,
    xyxy_to_cxcywh,
)

# Loss functions
from .blazeface_loss import (
    DetectionLoss,
    FocalLoss,
    SmoothL1Loss,
    MultiBoxLoss,
)

# Data loading
from .blazeface_dataloader import (
    BlazeFaceDataModule,
    DummyFaceDataset,
    collate_fn,
    get_default_transform,
)

# Training
from .blazeface_train import BlazeFaceLightningModule

# Inference
from .blazeface_inference import FaceDetector


__all__ = [
    # Model
    'BlazeFace',
    'BlazeBlock',
    'BlazeFaceBackbone',
    # Config
    'cfg_blazeface',
    'cfg_blazeface_front',
    'MATCHING_CONFIG',
    # Anchors
    'generate_anchors',
    'AnchorGenerator',
    'encode_boxes',
    'decode_boxes',
    'decode_keypoints',
    'compute_iou',
    'match_anchors',
    'anchors_to_xyxy',
    'xyxy_to_cxcywh',
    # Loss
    'DetectionLoss',
    'FocalLoss',
    'SmoothL1Loss',
    'MultiBoxLoss',
    # Data
    'BlazeFaceDataModule',
    'DummyFaceDataset',
    'collate_fn',
    'get_default_transform',
    # Training
    'BlazeFaceLightningModule',
    # Inference
    'FaceDetector',
]
