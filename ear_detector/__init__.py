# Ear Detector module - BlazeEar
# Following BlazeFace naming convention: blazeear.py, blazeear_anchors.py, blazeear_loss.py, etc.

from ear_detector.blazeear import BlazeEar, BlazeBlock, BlazeEarBackbone, create_blazeear
from ear_detector.lightning_module import BlazeEarLightningModule
from ear_detector.blazeear_dataloader import EarDetectorDataset, EarDetectorDataModule, collate_fn, get_default_transform
from ear_detector.blazeear_loss import DetectionLoss, FocalLoss, SmoothL1Loss
from ear_detector.blazeear_inference import EarDetector
from ear_detector.blazeear_anchors import (
    ANCHOR_CONFIG_16,
    ANCHOR_CONFIG_8,
    SCALE_FACTOR,
    VARIANCE,
    MATCHING_CONFIG,
    generate_anchors,
    encode_boxes,
    decode_boxes,
    match_anchors,
    compute_iou,
    anchors_to_xyxy,
    get_anchor_stats,
)
from ear_detector.config import cfg_blazeear, get_num_anchors_per_cell

__all__ = [
    # Model
    'BlazeEar',
    'BlazeBlock', 
    'BlazeEarBackbone',
    'create_blazeear',
    # Inference
    'EarDetector',
    # Training
    'BlazeEarLightningModule',
    'EarDetectorDataset',
    'EarDetectorDataModule',
    'collate_fn',
    'get_default_transform',
    # Loss
    'DetectionLoss',
    'FocalLoss',
    'SmoothL1Loss',
    # Config
    'cfg_blazeear',
    'get_num_anchors_per_cell',
    # Anchor utilities
    'ANCHOR_CONFIG_16',
    'ANCHOR_CONFIG_8',
    'SCALE_FACTOR',
    'VARIANCE',
    'MATCHING_CONFIG',
    'generate_anchors',
    'encode_boxes',
    'decode_boxes',
    'match_anchors',
    'compute_iou',
    'anchors_to_xyxy',
    'get_anchor_stats',
]
