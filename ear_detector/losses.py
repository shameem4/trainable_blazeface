"""Losses for BlazeEar detector.
Implements Focal Loss and Smooth L1 loss for detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops
from typing import Dict, Tuple, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in detection.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Args:
        alpha: Weighting factor for positive samples
        gamma: Focusing parameter
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N,) predicted logits
            targets: (N,) binary targets (0 or 1)
        """
        p = torch.sigmoid(predictions)
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for bounding box regression."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (N, 4) predicted box deltas
            targets: (N, 4) target box deltas
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes using torchvision.ops.box_iou.
    
    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format
        boxes2: (M, 4) boxes in [x1, y1, x2, y2] format
        
    Returns:
        (N, M) IoU matrix
    """
    return tv_ops.box_iou(boxes1, boxes2)


def encode_boxes(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
    scale: float = 128.0,
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors (BlazeFace-style).
    
    BlazeFace encoding (inverse of decoding):
        dx = (gt_cx - anchor_cx) / anchor_w * x_scale
        dy = (gt_cy - anchor_cy) / anchor_h * y_scale
        dw = gt_w / anchor_w * w_scale
        dh = gt_h / anchor_h * h_scale
    
    Since anchor_w = anchor_h = 1.0 for BlazeFace-style anchors:
        dx = (gt_cx - anchor_cx) * scale
        dw = gt_w * scale
    
    Args:
        gt_boxes: (N, 4) boxes in [x1, y1, x2, y2] normalized format
        anchors: (N, 4) anchors in [cx, cy, 1, 1] format
        scale: Scale factor (128 for BlazeFace front model)
        
    Returns:
        (N, 4) encoded targets
    """
    # Convert gt to center format
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    # BlazeFace-style encoding with scale factor
    dx = (gt_cx - anchors[:, 0]) * anchors[:, 2] * scale
    dy = (gt_cy - anchors[:, 1]) * anchors[:, 3] * scale
    dw = gt_w * anchors[:, 2] * scale
    dh = gt_h * anchors[:, 3] * scale
    
    return torch.stack([dx, dy, dw, dh], dim=-1)


class DetectionLoss(nn.Module):
    """
    Combined detection loss with focal loss for classification
    and smooth L1 for box regression.
    
    Args:
        pos_iou_threshold: IoU threshold for positive anchor matching
        neg_iou_threshold: IoU threshold for negative anchors
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        box_weight: Weight for box regression loss
        input_size: Input image size (for BlazeFace-style encoding)
    """
    
    def __init__(
        self,
        pos_iou_threshold: float = 0.1,  # Low threshold for BlazeFace-style unit anchors
        neg_iou_threshold: float = 0.05,  # Lower than pos to create ignore zone
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 0.01,
        input_size: int = 128,
    ):
        super().__init__()
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.box_weight = box_weight
        self.input_size = input_size  # For BlazeFace-style encoding
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.box_loss = SmoothL1Loss()
    
    def match_anchors(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match ground truth boxes to anchors.
        
        For BlazeFace-style unit anchors (w=h=1), we match by CENTER DISTANCE
        rather than IoU, since all unit anchors have similar IoU with small boxes.
        
        Strategy:
        1. Find closest anchor (by center distance) for each GT - always positive
        2. Additional anchors within distance threshold can also be positive
        
        Args:
            gt_boxes: (M, 4) ground truth boxes [x1, y1, x2, y2]
            anchors: (N, 4) anchors [cx, cy, w, h]
            
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
        
        # Compute GT box centers
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2  # (M,)
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2  # (M,)
        
        # Compute distance from each anchor center to each GT center
        # anchors: (N, 4) with [cx, cy, w, h]
        # Distance matrix: (N, M)
        anchor_cx = anchors[:, 0].unsqueeze(1)  # (N, 1)
        anchor_cy = anchors[:, 1].unsqueeze(1)  # (N, 1)
        gt_cx = gt_cx.unsqueeze(0)  # (1, M)
        gt_cy = gt_cy.unsqueeze(0)  # (1, M)
        
        distances = torch.sqrt((anchor_cx - gt_cx)**2 + (anchor_cy - gt_cy)**2)  # (N, M)
        
        # Find best (closest) GT for each anchor
        min_dist_per_anchor, best_gt_idx = distances.min(dim=1)  # (N,)
        
        # Initialize labels: all negative by default
        matched_labels = torch.zeros(num_anchors, dtype=torch.float32, device=device)
        
        # Find closest anchor for each GT (guaranteed positive)
        best_anchor_per_gt = distances.argmin(dim=0)  # (M,)
        
        # Mark closest anchors as positive
        for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
            matched_labels[anchor_idx] = 1
            best_gt_idx[anchor_idx] = gt_idx
        
        # Optionally: mark additional nearby anchors as positive
        # Using distance threshold based on anchor grid spacing
        # 16x16 grid: spacing = 1/16 = 0.0625, 8x8 grid: spacing = 1/8 = 0.125
        # Use threshold slightly larger than half the grid spacing
        pos_dist_threshold = self.pos_iou_threshold  # Reuse as distance threshold (~0.1)
        
        for gt_idx in range(gt_boxes.shape[0]):
            close_mask = distances[:, gt_idx] < pos_dist_threshold
            matched_labels[close_mask] = 1
            # Update best_gt_idx for newly marked positives
            best_gt_idx[close_mask] = gt_idx
        
        # Get matched boxes
        matched_boxes = gt_boxes[best_gt_idx]
        
        # Encode box targets with BlazeFace-style scaling
        matched_box_targets = encode_boxes(matched_boxes, anchors, scale=float(self.input_size))
        
        return matched_labels, matched_boxes, matched_box_targets
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        gt_boxes: List[torch.Tensor],
        anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Dict with 'classification' (B, N, 1) and 'box_regression' (B, N, 4)
            gt_boxes: List of (M, 4) tensors with ground truth boxes
            anchors: (N, 4) anchor boxes
            
        Returns:
            Dict with 'cls_loss', 'box_loss', 'total_loss'
        """
        batch_size = len(gt_boxes)
        device = predictions['classification'].device
        
        all_cls_preds = []
        all_cls_targets = []
        all_box_preds = []
        all_box_targets = []
        
        for i in range(batch_size):
            # Match anchors to ground truth
            matched_labels, _, matched_box_targets = self.match_anchors(
                gt_boxes[i].to(device),
                anchors,
            )
            
            # Get predictions for this image
            cls_pred = predictions['classification'][i, :, 0]
            box_pred = predictions['box_regression'][i]
            
            # Classification: use all non-ignored anchors
            valid_mask = matched_labels >= 0
            all_cls_preds.append(cls_pred[valid_mask])
            all_cls_targets.append(matched_labels[valid_mask])
            
            # Box regression: only positive anchors
            pos_mask = matched_labels == 1
            if pos_mask.sum() > 0:
                all_box_preds.append(box_pred[pos_mask])
                all_box_targets.append(matched_box_targets[pos_mask])
        
        # Compute losses
        cls_preds = torch.cat(all_cls_preds)
        cls_targets = torch.cat(all_cls_targets)
        cls_loss = self.focal_loss(cls_preds, cls_targets)
        
        if len(all_box_preds) > 0:
            box_preds = torch.cat(all_box_preds)
            box_targets = torch.cat(all_box_targets)
            box_loss = self.box_loss(box_preds, box_targets)
        else:
            box_loss = torch.tensor(0.0, device=device)
        
        total_loss = cls_loss + self.box_weight * box_loss
        
        return {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'total_loss': total_loss,
        }
