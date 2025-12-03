"""
Loss functions for ear detection and landmark models.

Provides loss classes for:
- Detection: Combined classification + regression loss
- Landmark: Keypoint localization loss
- Teacher: Combined detector + landmark loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DetectionLoss(nn.Module):
    """
    Loss function for object detection models (BlazeFace-style).
    
    Combines:
    - Classification loss (focal loss or BCE)
    - Regression loss (smooth L1 or IoU-based)
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        regression_weight: float = 1.0,
        use_focal_loss: bool = True
    ):
        """
        Args:
            num_classes: Number of object classes
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            regression_weight: Weight for regression loss
            use_focal_loss: Whether to use focal loss for classification
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.regression_weight = regression_weight
        self.use_focal_loss = use_focal_loss
        
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        Args:
            pred: Predicted logits [B, N, C]
            target: Target labels [B, N]
            
        Returns:
            Focal loss value
        """
        pred_sigmoid = torch.sigmoid(pred)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal weight
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()
    
    def regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regression loss for bounding box coordinates.
        
        Args:
            pred: Predicted coordinates [B, N, 4] (or more for keypoints)
            target: Target coordinates [B, N, 4]
            mask: Positive anchor mask [B, N]
            
        Returns:
            Regression loss value
        """
        # Only compute loss for positive anchors
        num_pos = mask.sum().clamp(min=1)
        
        loss = self.smooth_l1(pred, target)
        loss = (loss * mask.unsqueeze(-1)).sum() / num_pos
        
        return loss
    
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_coords: torch.Tensor,
        target_scores: torch.Tensor,
        target_coords: torch.Tensor,
        positive_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined detection loss.
        
        Args:
            pred_scores: Predicted classification scores [B, N, C]
            pred_coords: Predicted coordinates [B, N, coords]
            target_scores: Target classification labels [B, N, C]
            target_coords: Target coordinates [B, N, coords]
            positive_mask: Optional mask for positive anchors [B, N]
            
        Returns:
            Dictionary with 'classification', 'regression', and 'total' losses
        """
        if positive_mask is None:
            positive_mask = (target_scores.squeeze(-1) > 0.5).float()
        
        # Classification loss
        if self.use_focal_loss:
            cls_loss = self.focal_loss(pred_scores, target_scores)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(
                pred_scores, target_scores
            )
        
        # Regression loss
        reg_loss = self.regression_loss(
            pred_coords, target_coords, positive_mask
        )
        
        # Combined loss
        total_loss = cls_loss + self.regression_weight * reg_loss
        
        return {
            'classification': cls_loss,
            'regression': reg_loss,
            'total': total_loss
        }


class LandmarkLoss(nn.Module):
    """
    Loss function for landmark/keypoint detection models.
    
    Supports:
    - L1/L2/Smooth L1 regression losses
    - Wing loss for robust landmark detection
    """
    
    def __init__(
        self,
        num_keypoints: int = 55,
        loss_type: str = 'smooth_l1',
        wing_w: float = 10.0,
        wing_epsilon: float = 2.0
    ):
        """
        Args:
            num_keypoints: Number of keypoints to predict
            loss_type: Type of loss ('l1', 'l2', 'smooth_l1', 'wing')
            wing_w: Wing loss width parameter
            wing_epsilon: Wing loss epsilon parameter
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.loss_type = loss_type
        self.wing_w = wing_w
        self.wing_epsilon = wing_epsilon
        
        # Precompute wing loss constant
        self.wing_c = wing_w * (1 - torch.log(torch.tensor(1 + wing_w / wing_epsilon)))
    
    def wing_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute wing loss for landmark regression.
        
        Wing loss is more robust to small errors than L2, and handles
        large errors better than L1.
        
        Args:
            pred: Predicted keypoints [B, N*2] or [B, N, 2]
            target: Target keypoints [B, N*2] or [B, N, 2]
            
        Returns:
            Wing loss value
        """
        diff = torch.abs(pred - target)
        
        # Wing loss formula
        loss = torch.where(
            diff < self.wing_w,
            self.wing_w * torch.log(1 + diff / self.wing_epsilon),
            diff - self.wing_c.to(diff.device)
        )
        
        return loss.mean()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visibility: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute landmark loss.
        
        Args:
            pred: Predicted keypoints [B, N*2] or [B, N, 2]
            target: Target keypoints [B, N*2] or [B, N, 2]
            visibility: Optional visibility mask [B, N]
            
        Returns:
            Dictionary with 'landmark' and 'total' losses
        """
        # Flatten if needed
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_flat, target_flat, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_flat, target_flat, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_flat, target_flat, reduction='none')
        elif self.loss_type == 'wing':
            return {
                'landmark': self.wing_loss(pred_flat, target_flat),
                'total': self.wing_loss(pred_flat, target_flat)
            }
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply visibility mask if provided
        if visibility is not None:
            # Expand visibility to cover x,y coordinates
            vis_mask = visibility.unsqueeze(-1).expand(-1, -1, 2)
            vis_mask = vis_mask.reshape(vis_mask.shape[0], -1)
            loss = loss * vis_mask
            loss = loss.sum() / vis_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return {
            'landmark': loss,
            'total': loss
        }


class CombinedLoss(nn.Module):
    """
    Combined loss for models that do both detection and landmark prediction.
    """
    
    def __init__(
        self,
        detection_weight: float = 1.0,
        landmark_weight: float = 1.0,
        **kwargs
    ):
        """
        Args:
            detection_weight: Weight for detection loss
            landmark_weight: Weight for landmark loss
            **kwargs: Additional arguments for sub-losses
        """
        super().__init__()
        self.detection_weight = detection_weight
        self.landmark_weight = landmark_weight
        
        self.detection_loss = DetectionLoss(**kwargs.get('detection', {}))
        self.landmark_loss = LandmarkLoss(**kwargs.get('landmark', {}))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with 'scores', 'coords', 'keypoints'
            targets: Dict with 'scores', 'coords', 'keypoints', 'visibility'
            
        Returns:
            Dictionary with all loss components and total
        """
        losses = {}
        
        # Detection loss
        if 'scores' in predictions and 'coords' in predictions:
            det_losses = self.detection_loss(
                predictions['scores'],
                predictions['coords'],
                targets['scores'],
                targets['coords'],
                targets.get('positive_mask')
            )
            losses['det_classification'] = det_losses['classification']
            losses['det_regression'] = det_losses['regression']
            losses['detection'] = det_losses['total']
        
        # Landmark loss
        if 'keypoints' in predictions:
            lm_losses = self.landmark_loss(
                predictions['keypoints'],
                targets['keypoints'],
                targets.get('visibility')
            )
            losses['landmark'] = lm_losses['landmark']
        
        # Total loss
        total = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        if 'detection' in losses:
            total = total + self.detection_weight * losses['detection']
        if 'landmark' in losses:
            total = total + self.landmark_weight * losses['landmark']
        
        losses['total'] = total
        
        return losses


def get_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss by name.
    
    Args:
        loss_type: One of 'detection', 'landmark', 'combined'
        **kwargs: Arguments for the loss class
        
    Returns:
        Loss module
    """
    loss_map = {
        'detection': DetectionLoss,
        'landmark': LandmarkLoss,
        'combined': CombinedLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_map.keys())}")
    
    return loss_map[loss_type](**kwargs)
