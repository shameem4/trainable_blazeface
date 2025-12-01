"""
Losses for BlazeEar detector.
Implements Focal Loss and Smooth L1 loss for detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format
        boxes2: (M, 4) boxes in [x1, y1, x2, y2] format
        
    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    union = area1[:, None] + area2[None, :] - inter
    
    return inter / (union + 1e-6)


def encode_boxes(
    gt_boxes: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        gt_boxes: (N, 4) boxes in [x1, y1, x2, y2] format
        anchors: (N, 4) anchors in [cx, cy, w, h] format
        
    Returns:
        (N, 4) encoded deltas
    """
    # Convert gt to center format
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    # Encode
    dx = (gt_cx - anchors[:, 0]) / anchors[:, 2]
    dy = (gt_cy - anchors[:, 1]) / anchors[:, 3]
    dw = torch.log(gt_w / anchors[:, 2] + 1e-6)
    dh = torch.log(gt_h / anchors[:, 3] + 1e-6)
    
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
    """
    
    def __init__(
        self,
        pos_iou_threshold: float = 0.5,
        neg_iou_threshold: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 1.0,
    ):
        super().__init__()
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.box_weight = box_weight
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.box_loss = SmoothL1Loss()
    
    def match_anchors(
        self,
        gt_boxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Match ground truth boxes to anchors.
        
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
        
        # Convert anchors to corner format for IoU
        anchor_corners = torch.zeros_like(anchors)
        anchor_corners[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
        anchor_corners[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
        anchor_corners[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
        anchor_corners[:, 3] = anchors[:, 1] + anchors[:, 3] / 2
        
        # Compute IoU
        ious = compute_iou(anchor_corners, gt_boxes)  # (N, M)
        
        # Find best GT for each anchor
        best_iou, best_gt_idx = ious.max(dim=1)
        
        # Initialize labels
        matched_labels = torch.full((num_anchors,), -1, dtype=torch.float32, device=device)
        matched_labels[best_iou >= self.pos_iou_threshold] = 1
        matched_labels[best_iou < self.neg_iou_threshold] = 0
        
        # Get matched boxes
        matched_boxes = gt_boxes[best_gt_idx]
        
        # Encode box targets
        matched_box_targets = encode_boxes(matched_boxes, anchors)
        
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
