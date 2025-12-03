"""
Loss functions for ear detection models.

Provides:
- BlazeFaceDetectionLoss: Primary loss with BCE or focal loss option

Based on vincent1bt/blazeface-tensorflow loss implementation:
- Hard negative mining for background samples
- Huber/Smooth L1 for box regression
- BCE or Focal loss for classification
- Loss formula: detection_loss * 150 + background_loss * 35 + positive_loss * 35
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BlazeFaceDetectionLoss(nn.Module):
    """
    Loss function for BlazeFace detector following vincent1bt approach.
    
    Combines:
    - Classification loss (BCE or Focal loss with hard negative mining)
    - Regression loss (Huber/Smooth L1 for box coordinates)
    
    Key features:
    - Hard negative mining: Selects highest-scoring background anchors
    - Decodes anchor predictions to absolute coordinates for loss
    - Optional focal loss for better handling of class imbalance
    - Weight formula: detection * 150 + background * 35 + positive * 35
    """
    
    def __init__(
        self,
        hard_negative_ratio: int = 3,
        detection_weight: float = 150.0,
        classification_weight: float = 35.0,
        scale: int = 128,
        min_negatives_per_image: int = 10,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            hard_negative_ratio: Ratio of negatives to positives for hard mining
            detection_weight: Weight for box regression loss
            classification_weight: Weight for both positive and background loss
            scale: Image scale for decoding (128 for front model)
            min_negatives_per_image: Minimum number of negatives per image
            use_focal_loss: Whether to use focal loss instead of BCE
            focal_alpha: Focal loss alpha parameter (weight for positive class)
            focal_gamma: Focal loss gamma parameter (focusing parameter)
        """
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        self.scale = scale
        self.min_negatives_per_image = min_negatives_per_image
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Focal loss down-weights easy examples and focuses on hard negatives.
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
        
        Args:
            pred: Predicted probabilities (after sigmoid) [N]
            target: Target labels (0 or 1) [N]
            
        Returns:
            Focal loss value (scalar)
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy per element
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weight: (1 - pt)^gamma
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Alpha weighting
        alpha_weight = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()
    
    def bce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary cross entropy loss.
        
        Args:
            pred: Predicted probabilities (after sigmoid) [N]
            target: Target labels (0 or 1) [N]
            
        Returns:
            BCE loss value (scalar)
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        return bce.mean()
    
    def classification_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss (BCE or Focal).
        
        Args:
            pred: Predicted probabilities [N]
            target: Target labels [N]
            
        Returns:
            Classification loss value
        """
        if self.use_focal_loss:
            return self.focal_loss(pred, target)
        else:
            return self.bce_loss(pred, target)
    
    def decode_boxes(
        self,
        anchor_predictions: torch.Tensor,
        reference_anchors: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode anchor predictions to absolute box coordinates.

        Following vincent1bt/blazeface-tensorflow decoding (no anchor w/h scaling):
        - x_center = anchor_x + (pred_x / scale)
        - y_center = anchor_y + (pred_y / scale)
        - w = pred_w / scale
        - h = pred_h / scale

        Args:
            anchor_predictions: [B, 896, 4] predicted offsets [dx, dy, w, h]
            reference_anchors: [896, 2] or [896, 4] anchor centers [x, y, ...]

        Returns:
            [B, 896, 4] decoded boxes [x_min, y_min, x_max, y_max] in normalized coords
        """
        # Decode center and size (no anchor w/h multiplication - just scale division)
        x_center = reference_anchors[:, 0:1] + (anchor_predictions[..., 0:1] / self.scale)
        y_center = reference_anchors[:, 1:2] + (anchor_predictions[..., 1:2] / self.scale)

        w = anchor_predictions[..., 2:3] / self.scale
        h = anchor_predictions[..., 3:4] / self.scale

        # Convert to corners - [x_min, y_min, x_max, y_max] to match ground truth format
        y_min = y_center - h / 2
        x_min = x_center - w / 2
        y_max = y_center + h / 2
        x_max = x_center + w / 2

        return torch.cat([x_min, y_min, x_max, y_max], dim=-1)
    
    def forward(
        self,
        class_predictions: torch.Tensor,
        anchor_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        reference_anchors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BlazeFace detection loss.
        
        Following vincent1bt/blazeface-tensorflow loss_functions.py:
        - Hard negative mining for background
        - Huber loss for box regression (only positive anchors)
        - BCE for classification
        
        Args:
            class_predictions: [B, 896, 1] predicted class scores (sigmoid applied)
            anchor_predictions: [B, 896, 4] predicted box offsets [dx, dy, w, h]
            anchor_targets: [B, 896, 5] targets [class, x1, y1, x2, y2]
            reference_anchors: [896, 2] anchor centers [x, y]
            
        Returns:
            Dict with 'total', 'detection', 'background', 'positive' losses
        """
        B = class_predictions.shape[0]
        
        # Extract targets
        true_classes = anchor_targets[:, :, 0]  # [B, 896]
        true_coords = anchor_targets[:, :, 1:]  # [B, 896, 4]
        
        # Boolean mask for positive anchors
        faces_mask_bool = true_classes > 0.5  # [B, 896]
        
        # Count positives
        faces_num = faces_mask_bool.float().sum()
        
        # Hard negative mining: select ratio * num_positives negatives per batch
        # Select the highest-scoring background predictions
        background_num = max(
            int(faces_num * self.hard_negative_ratio) // B,
            self.min_negatives_per_image
        )
        
        # Squeeze class predictions
        class_pred_squeezed = class_predictions.squeeze(-1)  # [B, 896]
        
        # For hard negative mining: set positive locations to very low score
        # so they won't be selected as negatives
        predicted_classes_scores = torch.where(
            faces_mask_bool,
            torch.full_like(class_pred_squeezed, -99.0),
            class_pred_squeezed
        )  # [B, 896]
        
        # Sort and select top-k background predictions (hard negatives)
        sorted_scores, _ = torch.sort(predicted_classes_scores, dim=-1, descending=True)
        background_class_predictions = sorted_scores[:, :background_num]  # [B, background_num]
        
        # Get positive predictions
        positive_class_predictions = class_pred_squeezed[faces_mask_bool]  # [num_positives]
        
        # === Classification Loss ===
        # Background loss: predict 0 for background
        if background_class_predictions.numel() > 0:
            background_loss = self.classification_loss(
                background_class_predictions,
                torch.zeros_like(background_class_predictions)
            )
        else:
            background_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # Positive loss: predict 1 for faces/ears
        if positive_class_predictions.numel() > 0:
            positive_loss = self.classification_loss(
                positive_class_predictions,
                torch.ones_like(positive_class_predictions)
            )
        else:
            positive_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # === Regression Loss ===
        # Decode predictions to absolute coordinates
        offset_boxes = self.decode_boxes(anchor_predictions, reference_anchors)
        
        # Get predicted and true coords for positive anchors only
        filtered_pred_coords = offset_boxes[faces_mask_bool]  # [num_positives, 4]
        filtered_true_coords = true_coords[faces_mask_bool]   # [num_positives, 4]
        
        if filtered_pred_coords.numel() > 0:
            detection_loss = self.huber_loss(filtered_pred_coords, filtered_true_coords)
        else:
            detection_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # === Combined Loss ===
        # Following vincent1bt formula: detection * 150 + background * 35 + positive * 35
        total_loss = (
            detection_loss * self.detection_weight +
            background_loss * self.classification_weight +
            positive_loss * self.classification_weight
        )
        
        return {
            'total': total_loss,
            'detection': detection_loss,
            'background': background_loss,
            'positive': positive_loss,
            'num_positives': faces_num,
            'num_negatives': torch.tensor(background_num * B, device=class_predictions.device)
        }


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        box1: [N, 4] boxes in [x1, y1, x2, y2] format
        box2: [M, 4] boxes in [x1, y1, x2, y2] format

    Returns:
        [N, M] IoU matrix
    """
    # Format: [x1, y1, x2, y2]
    # Intersection
    x_min = torch.maximum(box1[:, None, 0], box2[None, :, 0])
    y_min = torch.maximum(box1[:, None, 1], box2[None, :, 1])
    x_max = torch.minimum(box1[:, None, 2], box2[None, :, 2])
    y_max = torch.minimum(box1[:, None, 3], box2[None, :, 3])

    intersection = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)

    # Union - area = (x2 - x1) * (y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection

    return intersection / (union + 1e-6)


def compute_mean_iou(
    pred_boxes: torch.Tensor,
    true_boxes: torch.Tensor,
    scale: float = 128.0
) -> torch.Tensor:
    """
    Compute mean IoU between predicted and true boxes (for metrics).
    
    Following vincent1bt approach: multiply by scale before computing IoU.
    
    Args:
        pred_boxes: [N, 4] predicted boxes in normalized [x1, y1, x2, y2]
        true_boxes: [N, 4] true boxes in normalized [x1, y1, x2, y2]
        scale: Scale factor (128 for front model)

    Returns:
        Mean IoU value
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Scale to pixel coordinates
    pred_scaled = pred_boxes * scale
    true_scaled = true_boxes * scale

    # Compute intersection - format: [x1, y1, x2, y2]
    x_min = torch.maximum(pred_scaled[:, 0], true_scaled[:, 0])
    y_min = torch.maximum(pred_scaled[:, 1], true_scaled[:, 1])
    x_max = torch.minimum(pred_scaled[:, 2], true_scaled[:, 2])
    y_max = torch.minimum(pred_scaled[:, 3], true_scaled[:, 3])

    # Add 1 like vincent1bt for pixel-based IoU
    intersection = torch.clamp(x_max - x_min + 1, min=0) * torch.clamp(y_max - y_min + 1, min=0)

    # Compute areas - area = (xmax - xmin + 1) * (ymax - ymin + 1)
    pred_area = (pred_scaled[:, 2] - pred_scaled[:, 0] + 1) * (pred_scaled[:, 3] - pred_scaled[:, 1] + 1)
    true_area = (true_scaled[:, 2] - true_scaled[:, 0] + 1) * (true_scaled[:, 3] - true_scaled[:, 1] + 1)
    
    union = pred_area + true_area - intersection
    
    iou = intersection / (union + 1e-6)
    
    return iou.mean()


def compute_map(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    true_boxes: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute mean Average Precision (mAP) for object detection.

    Simplified mAP calculation for single-class detection:
    - Sort predictions by confidence
    - Match predictions to ground truths by IoU
    - Compute precision at each recall level
    - Average precision across all ground truths

    Args:
        pred_boxes: [N, 4] predicted boxes [x1, y1, x2, y2] normalized
        pred_scores: [N] predicted confidence scores
        true_boxes: [M, 4] true boxes [x1, y1, x2, y2] normalized
        iou_threshold: IoU threshold for considering a match (default 0.5)

    Returns:
        Average Precision value
    """
    if pred_boxes.numel() == 0 or true_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[sorted_indices]

    num_gt = true_boxes.shape[0]
    gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device)

    tp = []  # True positives
    fp = []  # False positives

    # For each prediction (in order of confidence)
    for pred_box in pred_boxes_sorted:
        # Compute IoU with all ground truths
        ious = compute_iou_batch(pred_box.unsqueeze(0), true_boxes).squeeze(0)

        # Find best matching ground truth
        max_iou, max_idx = ious.max(dim=0)

        # Check if it's a match
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[max_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    if len(tp) == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    tp = torch.tensor(tp, dtype=torch.float32, device=pred_boxes.device)
    fp = torch.tensor(fp, dtype=torch.float32, device=pred_boxes.device)

    # Cumulative sums
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    # Precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)

    # Compute AP using 11-point interpolation
    ap = torch.zeros(1, device=pred_boxes.device)
    for t in torch.linspace(0, 1, 11, device=pred_boxes.device):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11

    return ap


def compute_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes [x1, y1, x2, y2]
        boxes2: [M, 4] boxes [x1, y1, x2, y2]

    Returns:
        [N, M] IoU matrix
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

    # Compute intersection - format: [x1, y1, x2, y2]
    x_min = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    y_min = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    x_max = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    y_max = torch.minimum(boxes1[..., 3], boxes2[..., 3])

    intersection = torch.clamp(x_max - x_min, min=0) * torch.clamp(y_max - y_min, min=0)

    # Compute areas - area = (x2 - x1) * (y2 - y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-6)

    return iou


def get_loss(**kwargs) -> nn.Module:
    """
    Factory function to get BlazeFace detection loss.

    Args:
        **kwargs: Arguments for BlazeFaceDetectionLoss
            - hard_negative_ratio: Ratio of negatives to positives (default: 3)
            - detection_weight: Weight for box regression (default: 150.0)
            - classification_weight: Weight for classification (default: 35.0)
            - scale: Image scale for decoding (default: 128)
            - use_focal_loss: Use focal loss instead of BCE (default: False)
            - focal_alpha: Focal loss alpha (default: 0.25)
            - focal_gamma: Focal loss gamma (default: 2.0)
        
    Returns:
        BlazeFaceDetectionLoss module
    """
    return BlazeFaceDetectionLoss(**kwargs)
