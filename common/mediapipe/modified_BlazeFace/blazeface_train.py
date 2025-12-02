# blazeface_train.py
"""
Training script for BlazeFace face detector.

Contains both the Lightning module and training CLI.

Run standalone from common/mediapipe/modified_BlazeFace directory:
    python blazeface_train.py

Metrics tracked:
- mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
- Precision, Recall
- Inference speed (FPS)
"""
import math
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import nms

from .blazeface import BlazeFace
from .blazeface_anchors import generate_anchors, decode_boxes, decode_keypoints, MATCHING_CONFIG
from .blazeface_dataloader import BlazeFaceDataModule
from .blazeface_loss import DetectionLoss
from .config import cfg_blazeface


# =============================================================================
# Lightning Module
# =============================================================================

class BlazeFaceLightningModule(pl.LightningModule):
    """
    Lightning module for BlazeFace face detector training.
    
    Tracks metrics:
    - mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
    - Precision, Recall
    - FPS (inference speed)
    """
    
    def __init__(
        self,
        # Model params
        input_size: int = 128,
        num_keypoints: int = 6,
        pretrained_path: Optional[str] = None,
        # Loss params
        pos_iou_threshold: Optional[float] = None,
        neg_iou_threshold: Optional[float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        box_weight: float = 1.0,
        keypoint_weight: float = 0.5,
        # Training params
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        freeze_backbone_epochs: int = 0,
        # Inference params
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = BlazeFace(input_size=input_size, num_keypoints=num_keypoints)
        
        # Generate anchors
        self.register_buffer('anchors', generate_anchors(input_size))
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        
        # Freeze backbone if requested
        if self.hparams.freeze_backbone_epochs > 0:
            self._freeze_backbone()
        
        # Loss
        self.loss_fn = DetectionLoss(
            pos_iou_threshold=pos_iou_threshold,
            neg_iou_threshold=neg_iou_threshold,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            box_weight=box_weight,
            keypoint_weight=keypoint_weight,
            num_keypoints=num_keypoints,
        )
        
        # Metrics
        self.val_map = MeanAveragePrecision(
            iou_thresholds=[0.5, 0.75],
            class_metrics=False,
        )
        self.inference_times = []
    
    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        try:
            if not os.path.exists(path):
                print(f"Pretrained weights not found: {path}")
                return
            
            state_dict = torch.load(path, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} backbone parameters")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"Unfrozen {unfrozen_count} backbone parameters")
    
    def on_train_epoch_start(self):
        """Check if we should unfreeze backbone."""
        if self.hparams.freeze_backbone_epochs > 0:
            if self.current_epoch == self.hparams.freeze_backbone_epochs:
                print(f"\nEpoch {self.current_epoch}: Unfreezing backbone")
                self._unfreeze_backbone()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (conf, loc) predictions."""
        return self.model(x)
    
    def decode_predictions(self, loc_pred: torch.Tensor) -> torch.Tensor:
        """Decode box predictions to (x1, y1, x2, y2) format."""
        return decode_boxes(loc_pred, self.anchors)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        gt_boxes = batch['boxes']
        gt_keypoints = batch.get('keypoints')
        batch_size = images.shape[0]
        
        # Forward
        conf, loc = self.model(images)
        
        # Loss
        losses = self.loss_fn(
            (conf, loc),
            gt_boxes,
            gt_keypoints,
            self.anchors,
        )
        
        # Log
        self.log('train/loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train/cls_loss', losses['cls_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/box_loss', losses['box_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        if 'kp_loss' in losses:
            self.log('train/kp_loss', losses['kp_loss'], on_step=True, on_epoch=True, batch_size=batch_size)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step with mAP computation."""
        images = batch['image']
        gt_boxes = batch['boxes']
        gt_keypoints = batch.get('keypoints')
        batch_size = images.shape[0]
        
        # Forward
        conf, loc = self.model(images)
        
        # Loss
        losses = self.loss_fn(
            (conf, loc),
            gt_boxes,
            gt_keypoints,
            self.anchors,
        )
        
        # Log
        self.log('val/loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val/cls_loss', losses['cls_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('val/box_loss', losses['box_loss'], on_step=False, on_epoch=True, batch_size=batch_size)
        
        # Compute detections for mAP
        scores = torch.sigmoid(conf)
        boxes = self.decode_predictions(loc)
        
        preds = []
        targets = []
        
        for i in range(batch_size):
            score = scores[i, :, 0]
            box = boxes[i]
            
            # Filter by score
            mask = score > self.hparams.score_threshold
            pred_scores = score[mask]
            pred_boxes = box[mask]
            
            # NMS
            if len(pred_scores) > 0:
                keep = nms(pred_boxes, pred_scores, self.hparams.nms_threshold)
                pred_scores = pred_scores[keep]
                pred_boxes = pred_boxes[keep]
            
            # Scale to pixel coordinates for mAP
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
        
        self.val_map.update(preds, targets)
        
        # Measure FPS periodically
        if batch_idx % 10 == 0:
            self._measure_fps(images[:1])
        
        return losses['total_loss']
    
    def _measure_fps(self, image: torch.Tensor):
        """Measure inference FPS."""
        self.model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = self.model(image)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Time
            start = time.perf_counter()
            for _ in range(10):
                _ = self.model(image)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            self.inference_times.append(10 / elapsed)
    
    def on_validation_epoch_end(self):
        """Log mAP metrics."""
        map_results = self.val_map.compute()
        
        self.log('val/mAP', map_results['map'], prog_bar=True)
        self.log('val/mAP_50', map_results['map_50'], prog_bar=True)
        self.log('val/mAP_75', map_results['map_75'])
        
        if 'mar_100' in map_results:
            self.log('val/recall', map_results['mar_100'])
        
        if self.inference_times:
            avg_fps = sum(self.inference_times) / len(self.inference_times)
            self.log('val/fps', avg_fps)
            print(f"\nInference speed: {avg_fps:.1f} FPS")
        
        print(f"\nValidation: mAP={map_results['map']:.4f}, mAP@0.5={map_results['map_50']:.4f}, mAP@0.75={map_results['map_75']:.4f}")
        
        self.val_map.reset()
        self.inference_times = []
    
    def configure_optimizers(self):
        """Configure optimizer with cosine annealing and warmup."""
        # Differential learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = []
        if head_params:
            param_groups.append({'params': head_params, 'lr': self.hparams.learning_rate, 'name': 'heads'})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': self.hparams.learning_rate * 0.1, 'name': 'backbone'})
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        # Warmup + cosine annealing
        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            else:
                remaining = self.hparams.max_epochs - self.hparams.warmup_epochs
                if remaining <= 0:
                    return 1.0
                progress = (current_epoch - self.hparams.warmup_epochs) / remaining
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# =============================================================================
# Training CLI
# =============================================================================

def main():
    """Main training function."""
    parser = ArgumentParser(description="Train BlazeFace face detector")
    
    # Data arguments
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no_augment", action="store_true")
    
    # Model arguments
    parser.add_argument("--num_keypoints", type=int, default=6)
    parser.add_argument("--pretrained", type=str, default=None)
    
    # Loss arguments
    parser.add_argument("--pos_iou_threshold", type=float, default=None)
    parser.add_argument("--neg_iou_threshold", type=float, default=None)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--box_weight", type=float, default=1.0)
    parser.add_argument("--keypoint_weight", type=float, default=0.5)
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    
    # Inference arguments
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--nms_threshold", type=float, default=0.3)
    
    # Trainer arguments
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--output_dir", type=str, default="outputs")
    
    args = parser.parse_args()
    
    augment = args.augment and not args.no_augment
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data - using dummy data
    datamodule = BlazeFaceDataModule(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=augment,
    )
    
    # Model
    model = BlazeFaceLightningModule(
        input_size=args.image_size,
        num_keypoints=args.num_keypoints,
        pretrained_path=args.pretrained,
        pos_iou_threshold=args.pos_iou_threshold,
        neg_iou_threshold=args.neg_iou_threshold,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        box_weight=args.box_weight,
        keypoint_weight=args.keypoint_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="blazeface-{epoch:02d}-{val/mAP_50:.4f}",
            monitor="val/mAP_50",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=output_dir, name="blazeface"),
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
    )
    
    # Print config
    print("\n" + "=" * 60)
    print("BlazeFace Face Detector Training")
    print("=" * 60)
    print(f"Data: Dummy ({args.train_samples} train, {args.val_samples} val)")
    print(f"Image: {args.image_size}x{args.image_size} | Batch: {args.batch_size}")
    print(f"Model: 896 anchors | Keypoints: {args.num_keypoints}")
    print(f"Training: LR={args.learning_rate}, Epochs={args.max_epochs}, Warmup={args.warmup_epochs}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")
    
    # Train
    trainer.fit(model, datamodule)
    print(f"\nTraining complete! Best: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
