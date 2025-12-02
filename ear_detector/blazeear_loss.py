"""Losses for BlazeEar detector.
Implements Focal Loss and Smooth L1 loss for detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from ear_detector.anchors import (
    VARIANCE,
    MATCHING_CONFIG,
    compute_iou,
    encode_boxes,
    match_anchors,
    anchors_to_xyxy,
)


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


# Note: compute_iou and encode_boxes are imported from ear_detector.anchors


class DetectionLoss(nn.Module):
    """
    Combined detection loss with focal loss for classification
    and smooth L1 for box regression.
    
    Uses IoU-based anchor matching with properly-sized anchors
    (ear-appropriate aspect ratios).
    
    Args:
        pos_iou_threshold: IoU threshold for positive anchor matching (default from MATCHING_CONFIG)
        neg_iou_threshold: IoU threshold for negative anchors (default from MATCHING_CONFIG)
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        box_weight: Weight for box regression loss
    """
    
    def __init__(
        self,
        pos_iou_threshold: Optional[float] = None,  # Use MATCHING_CONFIG if None
        neg_iou_threshold: Optional[float] = None,  # Use MATCHING_CONFIG if None
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 1.0,
    ):
        super().__init__()
        # Use centralized config as defaults
        self.pos_iou_threshold = pos_iou_threshold if pos_iou_threshold is not None else MATCHING_CONFIG['pos_iou_threshold']
        self.neg_iou_threshold = neg_iou_threshold if neg_iou_threshold is not None else MATCHING_CONFIG['neg_iou_threshold']
        self.box_weight = box_weight
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.box_loss = SmoothL1Loss()
    
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
            # Match anchors to ground truth using centralized function
            matched_labels, _, matched_box_targets = match_anchors(
                gt_boxes[i].to(device),
                anchors,
                pos_iou_threshold=self.pos_iou_threshold,
                neg_iou_threshold=self.neg_iou_threshold,
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
