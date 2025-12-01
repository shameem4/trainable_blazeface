"""
Anchor utilities for BlazeEar detector.

Centralizes anchor generation, encoding/decoding, and matching logic.
This module eliminates code duplication across model.py, losses.py, and anchor_data_view.py.
"""
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import torchvision.ops as tv_ops


# =============================================================================
# Anchor Configuration
# =============================================================================
# Based on data analysis of 17,447 ear bounding boxes:
#   - Width:  10th=0.015, 25th=0.027, 50th=0.049, 75th=0.086, 90th=0.130
#   - Height: 10th=0.033, 25th=0.059, 50th=0.101, 75th=0.161, 90th=0.225
#   - Aspect ratio (h/w): median=1.85, range 1.4-2.4

# 16x16 grid: smaller ears
ANCHOR_CONFIG_16 = {
    'base_sizes': [0.04, 0.08, 0.12],  # Cover 10th-75th percentile
    'aspect_ratios': [1.4, 1.85, 2.4],  # 25th, 50th, 75th percentile h/w
}

# 8x8 grid: larger ears  
ANCHOR_CONFIG_8 = {
    'base_sizes': [0.15, 0.25, 0.35],  # Cover 75th-95th percentile
    'aspect_ratios': [1.4, 1.85, 2.4],
}

# Encoding scale factor
SCALE_FACTOR = 128.0

# Matching thresholds for anchor-GT assignment
# These are used by DetectionLoss and should match visualization tools
MATCHING_CONFIG = {
    'pos_iou_threshold': 0.5,   # IoU >= this -> positive anchor
    'neg_iou_threshold': 0.4,   # IoU < this -> negative anchor
    # Anchors with IoU in [neg_threshold, pos_threshold) are ignored
    'min_anchor_iou': 0.3,      # Reject GT boxes where best anchor IoU < this
}


def get_num_anchors_per_cell(config: Dict) -> int:
    """Get number of anchors per grid cell for a config."""
    return len(config['base_sizes']) * len(config['aspect_ratios'])


# =============================================================================
# Anchor Generation
# =============================================================================

def generate_anchors(
    anchor_config_16: Dict = ANCHOR_CONFIG_16,
    anchor_config_8: Dict = ANCHOR_CONFIG_8,
) -> torch.Tensor:
    """
    Generate ear-specific anchors at two scales.
    
    Unlike BlazeFace's unit anchors (w=h=1), we use sized anchors with
    ear-appropriate aspect ratios (taller than wide, ~1.4-2.4 h/w ratio).
    
    This allows:
    1. Proper IoU-based anchor matching (unit anchors all have same IoU)
    2. Smaller regression targets (network predicts deltas, not absolute size)
    3. Better matching to ear shapes in the data
    
    Args:
        anchor_config_16: Config for 16x16 grid anchors
        anchor_config_8: Config for 8x8 grid anchors
        
    Returns:
        (N, 4) tensor of anchors in [cx, cy, w, h] normalized format
    """
    anchors = []
    
    # 16x16 grid anchors (smaller ears)
    for y in range(16):
        for x in range(16):
            cx = (x + 0.5) / 16
            cy = (y + 0.5) / 16
            for base_size in anchor_config_16['base_sizes']:
                for aspect_ratio in anchor_config_16['aspect_ratios']:
                    # aspect_ratio = h/w, so h = base * sqrt(ar), w = base / sqrt(ar)
                    w = base_size / (aspect_ratio ** 0.5)
                    h = base_size * (aspect_ratio ** 0.5)
                    anchors.append([cx, cy, w, h])
    
    # 8x8 grid anchors (larger ears)
    for y in range(8):
        for x in range(8):
            cx = (x + 0.5) / 8
            cy = (y + 0.5) / 8
            for base_size in anchor_config_8['base_sizes']:
                for aspect_ratio in anchor_config_8['aspect_ratios']:
                    w = base_size / (aspect_ratio ** 0.5)
                    h = base_size * (aspect_ratio ** 0.5)
                    anchors.append([cx, cy, w, h])
    
    return torch.tensor(anchors, dtype=torch.float32)


# =============================================================================
# Box Encoding/Decoding
# =============================================================================

def encode_boxes(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    scale: float = SCALE_FACTOR,
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors.
    
    Encoding formula:
        dx = (gt_cx - anchor_cx) / anchor_w * scale
        dy = (gt_cy - anchor_cy) / anchor_h * scale
        dw = gt_w / anchor_w * scale
        dh = gt_h / anchor_h * scale
    
    Args:
        gt_boxes: (N, 4) boxes in [x1, y1, x2, y2] normalized format
        anchors: (N, 4) anchors in [cx, cy, w, h] format
        scale: Scale factor for encoding
        
    Returns:
        (N, 4) encoded targets [dx, dy, dw, dh]
    """
    # Convert gt to center format
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    # Encode with anchor normalization
    dx = (gt_cx - anchors[:, 0]) / anchors[:, 2] * scale
    dy = (gt_cy - anchors[:, 1]) / anchors[:, 3] * scale
    dw = gt_w / anchors[:, 2] * scale
    dh = gt_h / anchors[:, 3] * scale
    
    return torch.stack([dx, dy, dw, dh], dim=-1)


def decode_boxes(
    box_regression: torch.Tensor,
    anchors: torch.Tensor,
    scale: float = SCALE_FACTOR,
) -> torch.Tensor:
    """
    Decode box regression predictions to actual boxes.
    
    Decoding formula (inverse of encoding):
        cx = dx / scale * anchor_w + anchor_cx
        cy = dy / scale * anchor_h + anchor_cy
        w = dw / scale * anchor_w
        h = dh / scale * anchor_h
    
    Args:
        box_regression: (..., 4) raw predictions [dx, dy, dw, dh]
        anchors: (N, 4) anchor boxes [cx, cy, w, h]
        scale: Scale factor (must match encoding)
        
    Returns:
        (..., 4) decoded boxes in [x1, y1, x2, y2] format
    """
    pred_cx = box_regression[..., 0] / scale * anchors[:, 2] + anchors[:, 0]
    pred_cy = box_regression[..., 1] / scale * anchors[:, 3] + anchors[:, 1]
    pred_w = box_regression[..., 2] / scale * anchors[:, 2]
    pred_h = box_regression[..., 3] / scale * anchors[:, 3]
    
    # Convert to corner format
    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2
    
    return torch.stack([x1, y1, x2, y2], dim=-1)


# =============================================================================
# Anchor Matching
# =============================================================================

def anchors_to_xyxy(anchors: torch.Tensor) -> torch.Tensor:
    """
    Convert anchors from [cx, cy, w, h] to [x1, y1, x2, y2] format.
    
    Args:
        anchors: (N, 4) anchors in [cx, cy, w, h] format
        
    Returns:
        (N, 4) anchors in [x1, y1, x2, y2] format
    """
    x1 = anchors[:, 0] - anchors[:, 2] / 2
    y1 = anchors[:, 1] - anchors[:, 3] / 2
    x2 = anchors[:, 0] + anchors[:, 2] / 2
    y2 = anchors[:, 1] + anchors[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format
        boxes2: (M, 4) boxes in [x1, y1, x2, y2] format
        
    Returns:
        (N, M) IoU matrix
    """
    return tv_ops.box_iou(boxes1, boxes2)


def match_anchors(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    pos_iou_threshold: float = 0.5,
    neg_iou_threshold: float = 0.4,
    scale: float = SCALE_FACTOR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match ground truth boxes to anchors using IoU.
    
    Strategy:
    1. Compute IoU between all anchors and GT boxes
    2. For each GT, find anchor with highest IoU - always positive
    3. Any anchor with IoU >= pos_threshold is positive
    4. Any anchor with IoU < neg_threshold is negative
    5. Anchors in between are ignored (don't contribute to loss)
    
    Args:
        gt_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
        anchors: (N, 4) anchors [cx, cy, w, h]
        pos_iou_threshold: IoU threshold for positive anchors
        neg_iou_threshold: IoU threshold for negative anchors
        scale: Scale factor for box encoding
        
    Returns:
        matched_labels: (N,) 1 for positive, 0 for negative, -1 for ignore
        matched_boxes: (N, 4) matched ground truth boxes
        matched_box_targets: (N, 4) encoded box targets
    """
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    if gt_boxes.shape[0] == 0:
        # No ground truth - all negative
        return (
            torch.zeros(num_anchors, device=device),
            torch.zeros((num_anchors, 4), device=device),
            torch.zeros((num_anchors, 4), device=device),
        )
    
    # Convert anchors to xyxy for IoU computation
    anchors_xyxy = anchors_to_xyxy(anchors)
    
    # Compute IoU matrix: (N, M)
    ious = compute_iou(anchors_xyxy, gt_boxes)
    
    # Find best GT for each anchor (highest IoU)
    max_iou_per_anchor, best_gt_idx = ious.max(dim=1)  # (N,)
    
    # Initialize labels: -1 for ignore (between neg and pos threshold)
    matched_labels = torch.full((num_anchors,), -1, dtype=torch.float32, device=device)
    
    # Negative: IoU < neg_threshold
    matched_labels[max_iou_per_anchor < neg_iou_threshold] = 0
    
    # Positive: IoU >= pos_threshold
    matched_labels[max_iou_per_anchor >= pos_iou_threshold] = 1
    
    # Ensure at least one anchor per GT is positive (highest IoU anchor)
    best_anchor_per_gt = ious.argmax(dim=0)  # (M,)
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
        matched_labels[anchor_idx] = 1
        best_gt_idx[anchor_idx] = gt_idx
    
    # Get matched boxes
    matched_boxes = gt_boxes[best_gt_idx]
    
    # Encode box targets
    matched_box_targets = encode_boxes(matched_boxes, anchors, scale=scale)
    
    return matched_labels, matched_boxes, matched_box_targets


# =============================================================================
# Visualization Helpers
# =============================================================================

def get_anchor_stats(anchors: torch.Tensor) -> Dict:
    """
    Get statistics about anchor sizes.
    
    Args:
        anchors: (N, 4) anchors in [cx, cy, w, h] format
        
    Returns:
        Dict with width/height min/max/unique values
    """
    widths = anchors[:, 2]
    heights = anchors[:, 3]
    
    return {
        'count': len(anchors),
        'width_min': widths.min().item(),
        'width_max': widths.max().item(),
        'height_min': heights.min().item(),
        'height_max': heights.max().item(),
        'unique_widths': sorted(set(widths.tolist())),
        'unique_heights': sorted(set(heights.tolist())),
        'aspect_ratios': sorted(set((heights / widths).tolist())),
    }
