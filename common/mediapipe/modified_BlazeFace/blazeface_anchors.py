# blazeface_anchors.py
"""
Anchor generation and utilities for BlazeFace.

Generates anchors for the two feature map scales (16x16 and 8x8).
Total anchors: 16*16*2 + 8*8*6 = 896

Also provides encoding/decoding functions for box regression.
"""
import torch
from itertools import product
from typing import Tuple, Optional

from .config import cfg_blazeface, MATCHING_CONFIG


# =============================================================================
# Anchor Generation
# =============================================================================

def generate_anchors(
    input_size: int = 128,
    cfg: dict = None,
) -> torch.Tensor:
    """
    Generate anchors for BlazeFace.
    
    Args:
        input_size: Input image size (default 128)
        cfg: Configuration dict (defaults to cfg_blazeface)
    
    Returns:
        Tensor of shape (896, 4) with anchors in (cx, cy, w, h) normalized format
    """
    if cfg is None:
        cfg = cfg_blazeface
    
    priors = []
    feature_maps = cfg['feature_maps']
    steps = cfg['steps']
    
    # Layer 1: 16x16 with 2 anchors per pixel
    anchor_sizes_16 = cfg['anchor_config_16']['base_sizes']
    f = feature_maps[0]
    step = steps[0]
    f_k = input_size / step
    
    for i, j in product(range(f[0]), range(f[1])):
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k
        
        for size in anchor_sizes_16:
            s = size / input_size
            priors.append([cx, cy, s, s])
    
    # Layer 2: 8x8 with 6 anchors per pixel
    anchor_sizes_8 = cfg['anchor_config_8']['base_sizes']
    f = feature_maps[1]
    step = steps[1]
    f_k = input_size / step
    
    for i, j in product(range(f[0]), range(f[1])):
        cx = (j + 0.5) / f_k
        cy = (i + 0.5) / f_k
        
        for size in anchor_sizes_8:
            s = size / input_size
            priors.append([cx, cy, s, s])
    
    output = torch.tensor(priors, dtype=torch.float32)
    
    if cfg.get('clip', False):
        output.clamp_(min=0, max=1)
    
    return output


class AnchorGenerator:
    """
    Legacy anchor generator class for backward compatibility.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.min_dim = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.clip = cfg.get('clip', False)

    def forward(self):
        return generate_anchors(self.min_dim, self.cfg)


# =============================================================================
# Anchor Format Conversion
# =============================================================================

def anchors_to_xyxy(anchors: torch.Tensor) -> torch.Tensor:
    """
    Convert anchors from (cx, cy, w, h) to (x1, y1, x2, y2) format.
    
    Args:
        anchors: Tensor of shape (N, 4) in (cx, cy, w, h) format
        
    Returns:
        Tensor of shape (N, 4) in (x1, y1, x2, y2) format
    """
    return torch.cat([
        anchors[:, :2] - anchors[:, 2:] / 2,  # x1, y1
        anchors[:, :2] + anchors[:, 2:] / 2,  # x2, y2
    ], dim=1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.
    
    Args:
        boxes: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape (N, 4) in (cx, cy, w, h) format
    """
    return torch.cat([
        (boxes[:, :2] + boxes[:, 2:]) / 2,  # cx, cy
        boxes[:, 2:] - boxes[:, :2],         # w, h
    ], dim=1)


# =============================================================================
# IoU Computation
# =============================================================================

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape (N, M) with IoU values
    """
    # Intersection
    max_xy = torch.min(
        boxes1[:, 2:].unsqueeze(1),  # (N, 1, 2)
        boxes2[:, 2:].unsqueeze(0),  # (1, M, 2)
    )
    min_xy = torch.max(
        boxes1[:, :2].unsqueeze(1),  # (N, 1, 2)
        boxes2[:, :2].unsqueeze(0),  # (1, M, 2)
    )
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]  # (N, M)
    
    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    
    # Union
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    return inter_area / (union + 1e-6)


# =============================================================================
# Encoding / Decoding
# =============================================================================

def encode_boxes(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    variance: Tuple[float, float] = None,
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        gt_boxes: Ground truth boxes (N, 4) in (x1, y1, x2, y2) format
        anchors: Anchor boxes (N, 4) in (cx, cy, w, h) format
        variance: Variance for encoding (default from MATCHING_CONFIG)
        
    Returns:
        Encoded boxes (N, 4) as (dx, dy, dw, dh)
    """
    if variance is None:
        variance = MATCHING_CONFIG['variance']
    
    # Convert GT to center format
    gt_cxcy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
    
    # Encode offsets
    delta_cxcy = (gt_cxcy - anchors[:, :2]) / (variance[0] * anchors[:, 2:])
    delta_wh = torch.log(gt_wh / anchors[:, 2:] + 1e-6) / variance[1]
    
    return torch.cat([delta_cxcy, delta_wh], dim=1)


def decode_boxes(
    predictions: torch.Tensor,
    anchors: torch.Tensor,
    variance: Tuple[float, float] = None,
) -> torch.Tensor:
    """
    Decode predicted box offsets to actual box coordinates.
    
    Args:
        predictions: Predicted offsets (N, 4) or (B, N, 4+)
        anchors: Anchor boxes (N, 4) in (cx, cy, w, h) format
        variance: Variance for decoding (default from MATCHING_CONFIG)
        
    Returns:
        Decoded boxes in (x1, y1, x2, y2) format
    """
    if variance is None:
        variance = MATCHING_CONFIG['variance']
    
    # Handle batch dimension
    if predictions.dim() == 3:
        # (B, N, 4+) -> process batch
        batch_size = predictions.size(0)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Decode center
    pred_cxcy = predictions[..., :2] * variance[0] * anchors[..., 2:] + anchors[..., :2]
    
    # Decode size
    pred_wh = torch.exp(predictions[..., 2:4] * variance[1]) * anchors[..., 2:]
    
    # Convert to corner format
    boxes = torch.cat([
        pred_cxcy - pred_wh / 2,  # x1, y1
        pred_cxcy + pred_wh / 2,  # x2, y2
    ], dim=-1)
    
    return boxes


def decode_keypoints(
    predictions: torch.Tensor,
    anchors: torch.Tensor,
    num_keypoints: int = 6,
    variance: Tuple[float, float] = None,
) -> torch.Tensor:
    """
    Decode predicted keypoint offsets.
    
    Args:
        predictions: Predicted offsets (N, 4 + num_keypoints*2) or (B, N, ...)
        anchors: Anchor boxes (N, 4) in (cx, cy, w, h) format
        num_keypoints: Number of keypoints
        variance: Variance for decoding
        
    Returns:
        Decoded keypoints (N, num_keypoints, 2) or (B, N, num_keypoints, 2)
    """
    if variance is None:
        variance = MATCHING_CONFIG['variance']
    
    # Handle batch dimension
    if predictions.dim() == 3:
        batch_size = predictions.size(0)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Extract keypoint predictions (after first 4 box values)
    kp_preds = predictions[..., 4:4 + num_keypoints * 2]
    
    # Reshape to (... , num_keypoints, 2)
    if predictions.dim() == 3:
        kp_preds = kp_preds.view(predictions.size(0), predictions.size(1), num_keypoints, 2)
        anchor_centers = anchors[..., :2].unsqueeze(2)
        anchor_sizes = anchors[..., 2:].unsqueeze(2)
    else:
        kp_preds = kp_preds.view(predictions.size(0), num_keypoints, 2)
        anchor_centers = anchors[:, :2].unsqueeze(1)
        anchor_sizes = anchors[:, 2:].unsqueeze(1)
    
    # Decode: offset from anchor center, scaled by anchor size
    keypoints = kp_preds * variance[0] * anchor_sizes + anchor_centers
    
    return keypoints


# =============================================================================
# Anchor Matching
# =============================================================================

def match_anchors(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    pos_threshold: float = None,
    neg_threshold: float = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match ground truth boxes to anchors.
    
    Args:
        gt_boxes: Ground truth boxes (M, 4) in (x1, y1, x2, y2) format
        anchors: Anchor boxes (N, 4) in (cx, cy, w, h) format
        pos_threshold: IoU threshold for positive match
        neg_threshold: IoU threshold for negative (background)
        
    Returns:
        matched_gt_idx: Index of matched GT for each anchor (N,), -1 for negative
        labels: Class label for each anchor (N,), 0=bg, 1=face
        ious: Best IoU for each anchor (N,)
    """
    if pos_threshold is None:
        pos_threshold = MATCHING_CONFIG['pos_iou_threshold']
    if neg_threshold is None:
        neg_threshold = MATCHING_CONFIG['neg_iou_threshold']
    
    num_anchors = anchors.size(0)
    num_gt = gt_boxes.size(0)
    
    if num_gt == 0:
        # No ground truth - all anchors are negative
        return (
            torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device),
            torch.zeros(num_anchors, dtype=torch.long, device=anchors.device),
            torch.zeros(num_anchors, dtype=torch.float32, device=anchors.device),
        )
    
    # Convert anchors to xyxy
    anchors_xyxy = anchors_to_xyxy(anchors)
    
    # Compute IoU
    iou = compute_iou(anchors_xyxy, gt_boxes)  # (N, M)
    
    # Best GT for each anchor
    best_gt_iou, best_gt_idx = iou.max(dim=1)  # (N,)
    
    # Assign labels
    labels = torch.zeros(num_anchors, dtype=torch.long, device=anchors.device)
    labels[best_gt_iou >= pos_threshold] = 1
    
    # Ignore zone (neither positive nor negative)
    ignore_mask = (best_gt_iou >= neg_threshold) & (best_gt_iou < pos_threshold)
    labels[ignore_mask] = -1  # Will be ignored in loss
    
    # Ensure each GT has at least one positive anchor
    best_anchor_for_gt, _ = iou.max(dim=0)  # (M,)
    for gt_idx in range(num_gt):
        best_anchor_idx = iou[:, gt_idx].argmax()
        labels[best_anchor_idx] = 1
        best_gt_idx[best_anchor_idx] = gt_idx
    
    # Set matched index to -1 for negatives
    matched_gt_idx = best_gt_idx.clone()
    matched_gt_idx[labels == 0] = -1
    
    return matched_gt_idx, labels, best_gt_iou