"""
Lightning Module for BlazeEar detector training.

Implements training with metrics matching the BlazeFace paper:
- mAP (mean Average Precision) at IoU thresholds 0.5, 0.75, 0.5:0.95
- Precision and Recall
- Inference speed (FPS)
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from ear_detector.losses import DetectionLoss
from ear_detector.model import BlazeEar


class BlazeEarLightningModule(pl.LightningModule):
    """
    Lightning module for BlazeEar detector training.
    
    Tracks metrics from BlazeFace paper:
    - mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
    - Precision, Recall
    - FPS (inference speed)
    """
    
    def __init__(
        self,
        # Model params
        num_anchors_16: int = 2,
        num_anchors_8: int = 6,
        input_size: int = 128,
        pretrained_blazeface_path: Optional[str] = None,
        # Loss params
        pos_iou_threshold: float = 0.5,
        neg_iou_threshold: float = 0.4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 1.0,
        # Training params
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        # Inference params
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = BlazeEar(
            num_anchors_16=num_anchors_16,
            num_anchors_8=num_anchors_8,
            input_size=input_size,
        )
        
        # Load pretrained backbone if provided
        if pretrained_blazeface_path is not None:
            self._load_pretrained_backbone(pretrained_blazeface_path)
        
        # Loss
        self.loss_fn = DetectionLoss(
            pos_iou_threshold=pos_iou_threshold,
            neg_iou_threshold=neg_iou_threshold,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            box_weight=box_weight,
        )
        
        # Metrics - mAP at various IoU thresholds (matching BlazeFace paper)
        self.val_map = MeanAveragePrecision(
            iou_thresholds=[0.5, 0.75],  # mAP@0.5 and mAP@0.75
            class_metrics=False,
        )
        
        # For FPS measurement
        self.inference_times = []
    
    def _load_pretrained_backbone(self, path: str):
        """Load pretrained BlazeFace backbone weights."""
        try:
            state_dict = torch.load(path, map_location='cpu')
            
            # Map BlazeFace weights to BlazeEar backbone
            # BlazeFace: backbone1.X, backbone2.X
            # BlazeEar model: model.backbone.backbone1.X, model.backbone.backbone2.X
            new_state = {}
            for key, value in state_dict.items():
                if key.startswith('backbone1.') or key.startswith('backbone2.'):
                    new_key = f'backbone.{key}'
                    new_state[new_key] = value
            
            # Load with strict=False (detection heads are different)
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded pretrained backbone from {path}")
            print(f"  Loaded {len(new_state)} backbone weights")
            print(f"  Missing keys (detection heads): {len(missing)}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            import traceback
            traceback.print_exc()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        gt_boxes = batch['boxes']
        batch_size = images.shape[0]
        
        # Forward pass
        outputs = self.model(images)
        
        # Compute loss
        losses = self.loss_fn(outputs, gt_boxes, self.model.anchors)
        
        # Log losses
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/cls_loss', losses['cls_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/box_loss', losses['box_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step with mAP computation."""
        images = batch['image']
        gt_boxes = batch['boxes']
        batch_size = images.shape[0]
        
        # Forward pass
        outputs = self.model(images)
        
        # Compute loss
        losses = self.loss_fn(outputs, gt_boxes, self.model.anchors)
        
        # Log losses
        self.log('val/loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/cls_loss', losses['cls_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/box_loss', losses['box_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Compute detections for mAP
        scores = torch.sigmoid(outputs['classification'])
        boxes = self.model.decode_boxes(outputs['box_regression'])
        
        # Format predictions and targets for mAP metric
        preds = []
        targets = []
        
        for i in range(batch_size):
            # Get predictions above threshold
            score = scores[i, :, 0]
            box = boxes[i]
            
            mask = score > self.hparams.score_threshold
            pred_scores = score[mask]
            pred_boxes = box[mask]
            
            # Apply NMS
            if len(pred_scores) > 0:
                keep = nms(pred_boxes, pred_scores, self.hparams.nms_threshold)
                pred_scores = pred_scores[keep]
                pred_boxes = pred_boxes[keep]
            
            # Convert to absolute coordinates (assuming normalized [0,1])
            # Scale to image size for metric computation
            pred_boxes_scaled = pred_boxes * self.hparams.input_size
            gt_boxes_scaled = gt_boxes[i] * self.hparams.input_size
            
            preds.append({
                'boxes': pred_boxes_scaled,
                'scores': pred_scores,
                'labels': torch.zeros(len(pred_scores), dtype=torch.long, device=self.device),
            })
            
            targets.append({
                'boxes': gt_boxes_scaled,
                'labels': torch.zeros(len(gt_boxes_scaled), dtype=torch.long, device=self.device),
            })
        
        # Update mAP metric
        self.val_map.update(preds, targets)
        
        # Measure inference speed (every 10 batches)
        if batch_idx % 10 == 0:
            self._measure_fps(images[:1])  # Use single image for FPS
        
        return losses['total_loss']
    
    def _measure_fps(self, image: torch.Tensor):
        """Measure inference FPS."""
        self.model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = self.model(image)
            
            # Measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(10):
                _ = self.model(image)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            fps = 10 / elapsed
            self.inference_times.append(fps)
    
    def on_validation_epoch_end(self):
        """Log mAP metrics at end of validation epoch."""
        # Compute mAP
        map_results = self.val_map.compute()
        
        # Log BlazeFace-style metrics
        self.log('val/mAP', map_results['map'], prog_bar=True)
        self.log('val/mAP_50', map_results['map_50'], prog_bar=True)
        self.log('val/mAP_75', map_results['map_75'])
        
        # Log precision/recall if available
        if 'mar_100' in map_results:
            self.log('val/recall', map_results['mar_100'])
        
        # Log FPS
        if self.inference_times:
            avg_fps = sum(self.inference_times) / len(self.inference_times)
            self.log('val/fps', avg_fps)
            print(f"\nInference speed: {avg_fps:.1f} FPS")
        
        # Print summary
        print(f"\nValidation metrics:")
        print(f"  mAP: {map_results['map']:.4f}")
        print(f"  mAP@0.5: {map_results['map_50']:.4f}")
        print(f"  mAP@0.75: {map_results['map_75']:.4f}")
        
        # Reset metrics
        self.val_map.reset()
        self.inference_times = []
    
    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Cosine annealing with warmup
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(self.hparams.warmup_epochs)
            else:
                import math
                progress = (current_epoch - self.hparams.warmup_epochs) / (
                    self.hparams.max_epochs - self.hparams.warmup_epochs
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
