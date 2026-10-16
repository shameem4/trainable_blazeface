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

from utils.metrics import compute_mean_iou_torch, compute_map_torch
from utils.box_utils import decode_boxes as _decode_boxes_util


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
        hard_negative_ratio: float = 1.0,
        detection_weight: float = 150.0,
        classification_weight: float = 35.0,
        scale: int = 128,
        min_negatives_per_image: int = 10,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        positive_classification_weight: Optional[float] = None,
        hard_negative_prior_cap: float = 1.0
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
            hard_negative_prior_cap: Ceiling on the share of the hard-negative budget
                any one anchor prior may take, as a fraction of the budget. 1.0 is a
                plain global top-k (the original behaviour). Below 1.0, each prior
                first offers only its own hardest ceil(cap * budget) candidates and
                the global top-k is drawn from that pool, so one prior cannot
                monopolise the negatives. Needs `prior_anchors` to be set, otherwise
                there is only one group and the cap does nothing.
            positive_classification_weight: Optional override for positive classification loss weight.
                Defaults to classification_weight when not provided.
        """
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.detection_weight = detection_weight
        self.classification_weight = classification_weight
        self.positive_classification_weight = \
            positive_classification_weight if positive_classification_weight is not None else classification_weight
        self.scale = scale
        self.min_negatives_per_image = min_negatives_per_image
        self.hard_negative_prior_cap = hard_negative_prior_cap
        self.prior_anchors = None   # set to the assignment priors to enable the cap
        self._prior_groups = None
        self._prior_groups_key = None
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
    
    def focal_loss_with_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss on raw logits (no clamp, so saturated predictions keep their gradient)."""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p = torch.sigmoid(logits)
        pt = target * p + (1 - target) * (1 - p)
        alpha_weight = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
        return (alpha_weight * (1 - pt) ** self.focal_gamma * bce).mean()

    def classification_loss_with_logits(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Classification loss (BCE or focal) computed from raw logits."""
        if self.use_focal_loss:
            return self.focal_loss_with_logits(logits, target)
        return F.binary_cross_entropy_with_logits(logits, target)

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

        Wrapper around utils.box_utils.decode_boxes.

        Args:
            anchor_predictions: [B, 896, 4] predicted offsets [dx, dy, w, h]
            reference_anchors: [896, 2] or [896, 4] anchor centers [x, y, ...]

        Returns:
            [B, 896, 4] decoded boxes [ymin, xmin, ymax, xmax] in normalized coords
        """
        return _decode_boxes_util(anchor_predictions, reference_anchors, scale=self.scale)
    
    def _groups_for(self, anchors: torch.Tensor) -> list:
        """Anchor indices grouped by (w, h) prior, cached per tensor identity."""
        key = (int(anchors.shape[0]), int(anchors.shape[1]), str(anchors.device))
        if getattr(self, "_prior_groups", None) is not None and self._prior_groups_key == key:
            return self._prior_groups
        if anchors.shape[1] < 4:
            groups = [torch.arange(anchors.shape[0], device=anchors.device)]
        else:
            wh = anchors[:, 2:4]
            unique = torch.unique(wh, dim=0)
            groups = [torch.nonzero((wh == row).all(dim=1), as_tuple=True)[0] for row in unique]
        self._prior_groups = groups
        self._prior_groups_key = key
        return groups

    def _mine_negatives(self, scores: torch.Tensor, budget: int) -> torch.Tensor:
        """The `budget` hardest background scores per image, [B, budget].

        With cap >= 1.0 this is a plain global top-k. Below 1.0 each prior offers
        only its own hardest candidates first, so no single prior can exceed the
        ceiling. The ceiling is one-sided: a prior whose logits are not competitive
        still contributes nothing, so this removes a concentration without
        manufacturing background pressure anywhere. The total stays `budget`, so the
        loss scale does not move.
        """
        if budget <= 0:
            return scores[:, :0]
        prior_anchors = getattr(self, "prior_anchors", None)
        if self.hard_negative_prior_cap >= 1.0 or prior_anchors is None:
            return torch.topk(scores, k=budget, dim=-1).values
        groups = self._groups_for(prior_anchors)
        if len(groups) < 2:
            return torch.topk(scores, k=budget, dim=-1).values
        per_group = max(int(-(-budget * self.hard_negative_prior_cap // 1)),
                        int(-(-budget // len(groups))), 1)
        floor_value = torch.finfo(scores.dtype).min
        offers = []
        for index in groups:
            take = min(per_group, int(index.numel()))
            group_top = torch.topk(scores.index_select(1, index), k=take, dim=-1).values
            if take < per_group:
                group_top = F.pad(group_top, (0, per_group - take), value=floor_value)
            offers.append(group_top)
        pool = torch.cat(offers, dim=-1)
        return torch.topk(pool, k=min(budget, pool.shape[-1]), dim=-1).values

    def forward(
        self,
        class_predictions: torch.Tensor,
        anchor_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        reference_anchors: torch.Tensor,
        class_logits: Optional[torch.Tensor] = None,
        anchor_ignore: Optional[torch.Tensor] = None
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
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax]
            reference_anchors: [896, 2] anchor centers [x, y]
            anchor_ignore: [B, 896] bool. Anchors overlapping a box that did not win
                a positive slot. Excluded from hard negative mining, so an anchor
                sitting on a real face at high IoU is not mined as background --
                mining otherwise seeks out exactly those high-scoring "negatives".
            class_logits: [B, 896, 1] raw logits. When given, classification losses are
                computed from logits (numerically stable, no gradient loss on saturated
                predictions) and class_predictions is only used for metrics/compat.
            
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
        use_logits = class_logits is not None
        class_pred_squeezed = (class_logits if use_logits else class_predictions).squeeze(-1)  # [B, 896]
        cls_loss = self.classification_loss_with_logits if use_logits else self.classification_loss
        
        # For hard negative mining: set positive locations to very low score
        # so they won't be selected as negatives
        # (-1e4 is below any logit and any probability, so positives are never selected)
        exclude_from_negatives = faces_mask_bool
        if anchor_ignore is not None:
            exclude_from_negatives = exclude_from_negatives | anchor_ignore.bool()

        predicted_classes_scores = torch.where(
            exclude_from_negatives,
            torch.full_like(class_pred_squeezed, -1e4),
            class_pred_squeezed
        )  # [B, 896]

        # Never mine more negatives than exist, or the pool refills with sentinels.
        available = int((~exclude_from_negatives).sum(dim=1).min().item())
        background_num = min(background_num, max(available, 0))

        background_class_predictions = self._mine_negatives(
            predicted_classes_scores, background_num)  # [B, background_num]
        
        # Get positive predictions
        positive_class_predictions = class_pred_squeezed[faces_mask_bool]  # [num_positives]
        
        # === Classification Loss ===
        # Background loss: predict 0 for background
        if background_class_predictions.numel() > 0:
            background_loss = cls_loss(
                background_class_predictions,
                torch.zeros_like(background_class_predictions)
            )
        else:
            background_loss = torch.tensor(0.0, device=class_predictions.device)
        
        # Positive loss: predict 1 for faces/ears
        if positive_class_predictions.numel() > 0:
            positive_loss = cls_loss(
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
            positive_loss * self.positive_classification_weight
        )
        
        return {
            'total': total_loss,
            'detection': detection_loss,
            'background': background_loss,
            'positive': positive_loss,
            'num_positives': faces_num,
            'num_negatives': torch.tensor(background_num * B, device=class_predictions.device)
        }


# Backwards compatibility re-exports (used in training scripts)
compute_mean_iou = compute_mean_iou_torch
compute_map = compute_map_torch


def get_loss(**kwargs) -> nn.Module:
    """
    Factory function to get BlazeFace detection loss.

    Args:
        **kwargs: Arguments for BlazeFaceDetectionLoss
            - hard_negative_ratio: Ratio of negatives to positives (default: 1.0)
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
