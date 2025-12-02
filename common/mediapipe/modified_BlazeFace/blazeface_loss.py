# blazeface_loss.py
"""
Losses for BlazeFace face detector.

Implements Focal Loss, Smooth L1 loss, and combined Detection Loss
for face detection with keypoints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional

from .blazeface_anchors import (
    MATCHING_CONFIG,
    compute_iou,
    encode_boxes,
    match_anchors,
    anchors_to_xyxy,
)


# =============================================================================
# Component Losses
# =============================================================================

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
    """Smooth L1 loss for bounding box and keypoint regression."""
    
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
            predictions: (N, D) predicted deltas
            targets: (N, D) target deltas
        """
        diff = torch.abs(predictions - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return loss.mean()


# =============================================================================
# Detection Loss
# =============================================================================

class DetectionLoss(nn.Module):
    """
    Combined detection loss with focal loss for classification,
    smooth L1 for box regression, and optional keypoint loss.
    
    Uses IoU-based anchor matching.
    
    Args:
        pos_iou_threshold: IoU threshold for positive anchor matching
        neg_iou_threshold: IoU threshold for negative anchors
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        box_weight: Weight for box regression loss
        keypoint_weight: Weight for keypoint regression loss
        num_keypoints: Number of keypoints (default 6 for BlazeFace)
    """
    
    def __init__(
        self,
        pos_iou_threshold: Optional[float] = None,
        neg_iou_threshold: Optional[float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 1.0,
        keypoint_weight: float = 0.5,
        num_keypoints: int = 6,
    ):
        super().__init__()
        # Use centralized config as defaults
        self.pos_iou_threshold = pos_iou_threshold if pos_iou_threshold is not None else MATCHING_CONFIG['pos_iou_threshold']
        self.neg_iou_threshold = neg_iou_threshold if neg_iou_threshold is not None else MATCHING_CONFIG['neg_iou_threshold']
        self.box_weight = box_weight
        self.keypoint_weight = keypoint_weight
        self.num_keypoints = num_keypoints
        
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.box_loss = SmoothL1Loss()
        self.keypoint_loss = SmoothL1Loss()
    
    def forward(
        self,
        predictions: Tuple[torch.Tensor, torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_keypoints: Optional[List[torch.Tensor]],
        anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Tuple of (conf, loc) from model
                - conf: (B, N, 1) classification logits
                - loc: (B, N, 4 + num_kp*2) box + keypoint regression
            gt_boxes: List of (M, 4) tensors with GT boxes (x1, y1, x2, y2)
            gt_keypoints: Optional list of (M, num_kp, 2) tensors with GT keypoints
            anchors: (N, 4) anchor boxes in (cx, cy, w, h) format
            
        Returns:
            Dict with 'cls_loss', 'box_loss', 'kp_loss', 'total_loss'
        """
        conf_pred, loc_pred = predictions
        batch_size = conf_pred.size(0)
        device = conf_pred.device
        
        all_cls_preds = []
        all_cls_targets = []
        all_box_preds = []
        all_box_targets = []
        all_kp_preds = []
        all_kp_targets = []
        
        for i in range(batch_size):
            # Match anchors to ground truth
            matched_gt_idx, labels, _ = match_anchors(
                gt_boxes[i].to(device),
                anchors,
                pos_threshold=self.pos_iou_threshold,
                neg_threshold=self.neg_iou_threshold,
            )
            
            # Get predictions for this image
            cls_pred = conf_pred[i, :, 0]  # (N,)
            box_pred = loc_pred[i, :, :4]   # (N, 4)
            
            # Classification: use all non-ignored anchors
            valid_mask = labels >= 0
            all_cls_preds.append(cls_pred[valid_mask])
            all_cls_targets.append(labels[valid_mask].float())
            
            # Box regression: only positive anchors
            pos_mask = labels == 1
            if pos_mask.sum() > 0:
                pos_box_pred = box_pred[pos_mask]
                
                # Get matched GT boxes and encode
                matched_boxes = gt_boxes[i][matched_gt_idx[pos_mask]]
                pos_anchors = anchors[pos_mask]
                encoded_targets = encode_boxes(matched_boxes, pos_anchors)
                
                all_box_preds.append(pos_box_pred)
                all_box_targets.append(encoded_targets)
                
                # Keypoints (if provided)
                if gt_keypoints is not None and gt_keypoints[i] is not None:
                    kp_pred = loc_pred[i, :, 4:4 + self.num_keypoints * 2]  # (N, num_kp*2)
                    pos_kp_pred = kp_pred[pos_mask]
                    
                    # Get matched keypoints and encode
                    matched_kps = gt_keypoints[i][matched_gt_idx[pos_mask]]  # (pos, num_kp, 2)
                    matched_kps_flat = matched_kps.view(-1, self.num_keypoints * 2)
                    
                    # Encode keypoints relative to anchors
                    variance = MATCHING_CONFIG['variance']
                    anchor_centers = pos_anchors[:, :2]  # (pos, 2)
                    anchor_sizes = pos_anchors[:, 2:]    # (pos, 2)
                    
                    # Encode each keypoint
                    encoded_kps = []
                    for kp_idx in range(self.num_keypoints):
                        kp_xy = matched_kps[:, kp_idx, :]  # (pos, 2)
                        encoded_kp = (kp_xy - anchor_centers) / (variance[0] * anchor_sizes)
                        encoded_kps.append(encoded_kp)
                    encoded_kp_targets = torch.cat(encoded_kps, dim=1)  # (pos, num_kp*2)
                    
                    all_kp_preds.append(pos_kp_pred)
                    all_kp_targets.append(encoded_kp_targets)
        
        # Compute classification loss
        if len(all_cls_preds) > 0:
            cls_preds = torch.cat(all_cls_preds)
            cls_targets = torch.cat(all_cls_targets)
            cls_loss = self.focal_loss(cls_preds, cls_targets)
        else:
            cls_loss = torch.tensor(0.0, device=device)
        
        # Compute box loss
        if len(all_box_preds) > 0:
            box_preds = torch.cat(all_box_preds)
            box_targets = torch.cat(all_box_targets)
            box_loss = self.box_loss(box_preds, box_targets)
        else:
            box_loss = torch.tensor(0.0, device=device)
        
        # Compute keypoint loss
        if len(all_kp_preds) > 0:
            kp_preds = torch.cat(all_kp_preds)
            kp_targets = torch.cat(all_kp_targets)
            kp_loss = self.keypoint_loss(kp_preds, kp_targets)
        else:
            kp_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = cls_loss + self.box_weight * box_loss + self.keypoint_weight * kp_loss
        
        return {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'kp_loss': kp_loss,
            'total_loss': total_loss,
        }


# =============================================================================
# Legacy MultiBoxLoss (for backward compatibility)
# =============================================================================

def jaccard(box_a, box_b):
    """Compute IoU between two sets of boxes (legacy function)."""
    return compute_iou(box_a, box_b)


def intersect(box_a, box_b):
    """Compute intersection between two sets of boxes (legacy function)."""
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
                       box_b[:, 2:].unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
                       box_b[:, :2].unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def point_form(boxes):
    """Convert (cx, cy, w, h) to (xmin, ymin, xmax, ymax) - legacy function."""
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)


class MultiBoxLoss(nn.Module):
    """Legacy MultiBoxLoss for backward compatibility. Use DetectionLoss instead."""
    
    def __init__(self, cfg, overlap_thresh=0.35, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.variance = cfg.get('variance', [0.1, 0.2])
        self.threshold = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.num_keypoints = cfg.get('num_keypoints', 6)
        
        # Use new DetectionLoss internally
        self._detection_loss = DetectionLoss(
            pos_iou_threshold=overlap_thresh,
            neg_iou_threshold=overlap_thresh,
            num_keypoints=self.num_keypoints,
        )
    
    def forward(self, predictions, targets, priors):
        """
        Legacy forward pass.
        
        Args:
            predictions: tuple (conf_data, loc_data)
            targets: list of [num_objs, 4 + num_kp*2] tensors
            priors: [num_priors, 4] (cx, cy, w, h)
        """
        conf_data, loc_data = predictions
        
        # Extract boxes and keypoints from targets
        gt_boxes = []
        gt_keypoints = []
        for target in targets:
            boxes = target[:, :4]  # (M, 4) - assuming xyxy format
            gt_boxes.append(boxes)
            
            if target.size(1) > 4:
                kps = target[:, 4:].view(-1, self.num_keypoints, 2)
                gt_keypoints.append(kps)
            else:
                gt_keypoints.append(None)
        
        # Call new loss
        losses = self._detection_loss(
            (conf_data, loc_data),
            gt_boxes,
            gt_keypoints if any(k is not None for k in gt_keypoints) else None,
            priors,
        )
        
        return losses['box_loss'] + losses.get('kp_loss', 0), losses['cls_loss']