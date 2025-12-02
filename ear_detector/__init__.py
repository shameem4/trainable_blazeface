# Ear Detector module - BlazeEar
from ear_detector.model import BlazeEar, BlazeBlock, BlazeEarBackbone, create_blazeear
from ear_detector.lightning_module import BlazeEarLightningModule
from ear_detector.dataset import EarDetectorDataset, collate_fn, get_default_transform
from ear_detector.datamodule import EarDetectorDataModule
from ear_detector.losses import DetectionLoss, FocalLoss, SmoothL1Loss
from ear_detector.anchors import (
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

__all__ = [
    'BlazeEar',
    'BlazeBlock', 
    'BlazeEarBackbone',
    'create_blazeear',
    'BlazeEarLightningModule',
    'EarDetectorDataset',
    'EarDetectorDataModule',
    'collate_fn',
    'get_default_transform',
    'DetectionLoss',
    'FocalLoss',
    'SmoothL1Loss',
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
