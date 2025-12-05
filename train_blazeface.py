"""
Training script for BlazeFace Face detector.

Complete training pipeline following vincent1bt/blazeface-tensorflow methodology:
- Anchor-based target encoding (from dataloader)
- Hard negative mining loss
- BCE or Focal loss for classification
- Huber loss for box regression

Usage (NPY format):
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy --val-data data/preprocessed/val_detector.npy
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy --epochs 500 --lr 1e-4

Usage (CSV format):
    # Default: MediaPipe weight initialization with auto-resume
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface

    # Train from scratch (random initialization)
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface --init-weights scratch

    # Start fresh (disable auto-resume, but use MediaPipe weights)
    python train_blazeface.py --csv-format --train-data data/splits/train.csv --val-data data/splits/val.csv --data-root data/raw/blazeface --no-auto-resume
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from blazeface import BlazeFace
from blazebase import generate_reference_anchors, load_mediapipe_weights
from dataloader import create_dataloader
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou, compute_map


class BlazeFaceTrainer:
    """
    Trainer for BlazeFace Face detector.
    
    Handles training loop, validation, checkpointing, and logging.
    Following vincent1bt methodology for loss computation and metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        model_name: str = 'BlazeFace',
        scale: int = 128
    ):
        """
        Args:
            model: BlazeFace model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            loss_fn: Loss function (BlazeFaceDetectionLoss if None)
            optimizer: Optimizer (AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            model_name: Name for saving checkpoints
            scale: Image scale for decoding (128 for front, 256 for back)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.scale = scale
        
        # Generate reference anchors for loss computation
        # generate_reference_anchors returns (reference_anchors, small, big) tuple
        reference_anchors, _, _ = generate_reference_anchors()
        self.reference_anchors = reference_anchors.float().to(device)
        
        # Setup loss function
        self.loss_fn = loss_fn if loss_fn else BlazeFaceDetectionLoss(scale=scale)
        self.loss_fn = self.loss_fn.to(device)
        
        # Setup optimizer
        self.optimizer = optimizer if optimizer else optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        self.scheduler = scheduler
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = SummaryWriter(self.log_dir / model_name)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.metrics = {
            'positive_correct': 0,
            'positive_total': 0,
            'background_correct': 0,
            'background_total': 0
        }
    
    def _get_training_outputs(self, images: torch.Tensor) -> tuple:
        """
        Get raw training outputs from BlazeFace model.

        BlazeFace model returns (raw_boxes, raw_scores) from get_training_outputs()
        for training, which bypasses post-processing.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            class_predictions: [B, 896, 1] sigmoid scores (probabilities)
            anchor_predictions: [B, 896, 4] box predictions
        """
        # Use training output method that bypasses NMS
        if hasattr(self.model, 'get_training_outputs'):
            raw_boxes, raw_scores = self.model.get_training_outputs(images)
            # raw_boxes: [B, 896, 16] or [B, 896, 4]
            # raw_scores: [B, 896, 1] - raw logits

            # Extract first 4 coords if model outputs 16 (for keypoints)
            if raw_boxes.shape[-1] > 4:
                raw_boxes = raw_boxes[..., :4]

            # Apply sigmoid to convert logits to probabilities
            # Loss function expects probabilities, not logits
            class_predictions = torch.sigmoid(raw_scores)

            return class_predictions, raw_boxes
        else:
            # Fallback: call model directly
            output = self.model(images)
            if isinstance(output, tuple):
                scores = torch.sigmoid(output[1])  # Apply sigmoid
                return scores, output[0]  # scores, boxes
            raise ValueError("Model must have get_training_outputs() method")
    
    def _compute_metrics(
        self,
        class_predictions: torch.Tensor,
        anchor_targets: torch.Tensor,
        anchor_predictions: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute training metrics following vincent1bt.

        Metrics:
        - Positive accuracy: % of positive anchors correctly classified
        - Background accuracy: % of background anchors correctly classified
        - Mean IoU: Average IoU for positive predictions
        - mAP@0.5: Mean Average Precision at IoU threshold 0.5

        Args:
            class_predictions: [B, 896, 1] predicted scores
            anchor_targets: [B, 896, 5] targets [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
            anchor_predictions: [B, 896, 4] predicted boxes
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        true_classes = anchor_targets[:, :, 0]  # [B, 896]
        true_coords = anchor_targets[:, :, 1:]  # [B, 896, 4]
        pred_scores = class_predictions.squeeze(-1)  # [B, 896]

        # Positive accuracy
        positive_mask = true_classes > 0.5
        if positive_mask.sum() > 0:
            positive_preds = pred_scores[positive_mask] > threshold
            positive_acc = positive_preds.float().mean().item()
        else:
            positive_acc = 0.0

        # Background accuracy
        background_mask = true_classes < 0.5
        if background_mask.sum() > 0:
            background_preds = pred_scores[background_mask] < threshold
            background_acc = background_preds.float().mean().item()
        else:
            background_acc = 1.0

        # Mean IoU and mAP for positive predictions
        mean_iou = 0.0
        map_50 = 0.0

        if positive_mask.sum() > 0:
            # Decode predictions
            decoded_boxes = self.loss_fn.decode_boxes(
                anchor_predictions, self.reference_anchors
            )

            # Mean IoU for positive anchors
            pred_coords = decoded_boxes[positive_mask]
            gt_coords = true_coords[positive_mask]
            mean_iou = compute_mean_iou(pred_coords, gt_coords, scale=self.scale).item()

            # Compute mAP per batch
            batch_size = class_predictions.shape[0]
            map_scores = []

            for b in range(batch_size):
                # Get positive predictions (confidence > threshold)
                batch_scores = pred_scores[b]
                batch_decoded = decoded_boxes[b]
                batch_gt_mask = true_classes[b] > 0.5

                if batch_gt_mask.sum() == 0:
                    continue

                pred_mask = batch_scores > threshold
                if pred_mask.sum() == 0:
                    map_scores.append(0.0)
                    continue

                # Get predictions and ground truths
                pred_boxes_batch = batch_decoded[pred_mask]
                pred_scores_batch = batch_scores[pred_mask]
                gt_boxes_batch = true_coords[b][batch_gt_mask]

                # Compute mAP for this batch
                ap = compute_map(pred_boxes_batch, pred_scores_batch, gt_boxes_batch, iou_threshold=0.5)
                map_scores.append(ap.item())

            if map_scores:
                map_50 = sum(map_scores) / len(map_scores)

        return {
            'positive_acc': positive_acc,
            'background_acc': background_acc,
            'mean_iou': mean_iou,
            'map_50': map_50
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses and metrics for the epoch
        """
        self.model.train()
        epoch_losses = {}
        epoch_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'mean_iou': 0.0, 'map_50': 0.0}
        num_batches = 0
        
        batch_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            anchor_targets = batch['anchor_targets'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            class_predictions, anchor_predictions = self._get_training_outputs(images)
            
            # Compute loss
            losses = self.loss_fn(
                class_predictions,
                anchor_predictions,
                anchor_targets,
                self.reference_anchors
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                metrics = self._compute_metrics(
                    class_predictions, anchor_targets, anchor_predictions
                )
            
            # Accumulate losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item()
            
            # Accumulate metrics
            for key, value in metrics.items():
                epoch_metrics[key] += value
            
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % 10 == 0:
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
                for key, value in metrics.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
            
            # Print progress (following vincent1bt style)
            if batch_idx % 20 == 0 or batch_idx == len(self.train_loader) - 1:
                step_time = time.time() - batch_time
                print(f'\r  Step {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {losses["total"].item():.5f} | '
                      f'Pos Acc: {metrics["positive_acc"]:.4f} | '
                      f'Bg Acc: {metrics["background_acc"]:.4f} | '
                      f'IoU: {metrics["mean_iou"]:.4f} | '
                      f'mAP: {metrics["map_50"]:.4f} | '
                      f'Time: {step_time:.2f}s', end='')
                batch_time = time.time()
        
        print()  # New line after epoch
        
        # Average losses and metrics
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Combine into single dict
        epoch_losses.update(epoch_metrics)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of average validation losses and metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {}
        val_metrics = {'positive_acc': 0.0, 'background_acc': 0.0, 'mean_iou': 0.0, 'map_50': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                anchor_targets = batch['anchor_targets'].to(self.device)
                
                class_predictions, anchor_predictions = self._get_training_outputs(images)
                
                losses = self.loss_fn(
                    class_predictions,
                    anchor_predictions,
                    anchor_targets,
                    self.reference_anchors
                )
                
                metrics = self._compute_metrics(
                    class_predictions, anchor_targets, anchor_predictions
                )
                
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        if key not in val_losses:
                            val_losses[key] = 0.0
                        val_losses[key] += value.item()
                
                for key, value in metrics.items():
                    val_metrics[key] += value
                
                num_batches += 1
        
        # Average
        for key in val_losses:
            val_losses[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_losses[key], self.global_step)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_metrics[key], self.global_step)
        
        val_losses.update(val_metrics)
        
        return val_losses
    
    def save_checkpoint(self, filename: Optional[str] = None, is_best: bool = False):
        """Save training checkpoint."""
        if filename is None:
            filename = f'{self.model_name}_epoch{self.epoch}.pth'
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f'  Saved checkpoint: {path}')
        
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f'  Saved best model: {best_path}')
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f'Loaded checkpoint from epoch {self.epoch}')
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1
    ):
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Run validation every N epochs
        """
        print(f'\nStarting training for {num_epochs} epochs')
        print(f'Device: {self.device}')
        print(f'Training samples: {len(self.train_loader.dataset)}')
        if self.val_loader:
            print(f'Validation samples: {len(self.val_loader.dataset)}')
        print('-' * 60)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            epoch_str = f'{epoch + 1:03d}/{start_epoch + num_epochs}'
            print(f'\nEpoch {epoch_str}')
            
            # Train
            train_results = self.train_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f'  Train | Loss: {train_results["total"]:.5f} | '
                  f'Pos Acc: {train_results["positive_acc"]:.4f} | '
                  f'Bg Acc: {train_results["background_acc"]:.4f} | '
                  f'IoU: {train_results["mean_iou"]:.4f} | '
                  f'mAP: {train_results["map_50"]:.4f} | '
                  f'Time: {epoch_time:.1f}s')
            
            # Validate
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_results = self.validate()
                print(f'  Val   | Loss: {val_results["total"]:.5f} | '
                      f'Pos Acc: {val_results["positive_acc"]:.4f} | '
                      f'Bg Acc: {val_results["background_acc"]:.4f} | '
                      f'IoU: {val_results["mean_iou"]:.4f} | '
                      f'mAP: {val_results["map_50"]:.4f}')
                
                # Check for best model
                if val_results['total'] < self.best_val_loss:
                    self.best_val_loss = val_results['total']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
        
        # Save final checkpoint
        self.save_checkpoint(f'{self.model_name}_final.pth')
        self.writer.close()
        
        print('\n' + '=' * 60)
        print('Training complete!')
        print(f'Best validation loss: {self.best_val_loss:.5f}')
        print(f'Checkpoints saved to: {self.checkpoint_dir}')
        print('=' * 60)


def create_model(
    init_weights: str = 'mediapipe',
    weights_path: str = 'model_weights/blazeface.pth'
) -> BlazeFace:
    """
    Create BlazeFace model with specified weight initialization.

    Args:
        init_weights: Weight initialization strategy:
            - 'scratch': Random initialization
            - 'mediapipe': Load MediaPipe pretrained weights (default)
        weights_path: Path to MediaPipe weights file

    Returns:
        BlazeFace model
    """
    model = BlazeFace()

    if init_weights == 'mediapipe':
        weights_path = Path(weights_path)
        if weights_path.exists():
            print(f'Loading MediaPipe weights from {weights_path}')
            missing, unexpected = load_mediapipe_weights(model, str(weights_path), strict=False)
            if missing:
                print(f'  Missing keys: {len(missing)}')
            if unexpected:
                print(f'  Unexpected keys: {len(unexpected)}')
            print('  Successfully loaded MediaPipe weights (backbone + detection heads)')
        else:
            print(f'Warning: MediaPipe weights not found at {weights_path}')
            print('         Using random initialization instead')
            init_weights = 'scratch'

    if init_weights == 'scratch':
        print('Using random weight initialization')

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train BlazeFace face detector',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-data', type=str, default="data/splits/train_new.csv",
                        help='Path to training CSV file')
    parser.add_argument('--val-data', type=str, default="data/splits/val_new.csv",
                        help='Path to validation CSV file')
    parser.add_argument('--data-root', type=str, default="data/raw/blazeface",
                        help='Root directory for image paths (required for CSV)')
    
    # Model arguments
    parser.add_argument('--init-weights', type=str, default='mediapipe',
                        choices=['scratch', 'mediapipe'],
                        help='Weight initialization: scratch (random) or mediapipe (pretrained)')
    parser.add_argument('--weights-path', type=str, default='model_weights/blazeface.pth',
                        help='Path to MediaPipe weights file (used with --init-weights=mediapipe)')
    parser.add_argument('--no-freeze-keypoint-heads', action='store_true',
                        help='Allow keypoint regressors to update (default: frozen)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (vincent1bt uses 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true', 
                        help='Use focal loss instead of BCE')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')
    parser.add_argument('--detection-weight', type=float, default=150.0,
                        help='Weight for detection/regression loss')
    parser.add_argument('--classification-weight', type=float, default=35.0,
                        help='Weight for classification loss')
    parser.add_argument('--hard-negative-ratio', type=int, default=3,
                        help='Ratio of negatives to positives in hard mining')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='runs/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='runs/logs',
                        help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, 
                        default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'
    
    # Input size (fixed at 128x128 for front model)
    target_size = (128, 128)
    scale = 128
    
    print('=' * 60)
    print('BlazeFace Face Detector Training')
    print('=' * 60)
    print(f'Input size: {target_size}')
    print(f'Device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Epochs: {args.epochs}')
    print(f'Loss: {"Focal" if args.use_focal_loss else "BCE"} + Huber')
    print(f'Loss weights: detection={args.detection_weight}, cls={args.classification_weight}')
    print('=' * 60)
    
    # Create model with requested initialization
    model = create_model(
        init_weights=args.init_weights,
        weights_path=args.weights_path
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    freeze_kp = not args.no_freeze_keypoint_heads
    if freeze_kp and hasattr(model, "freeze_keypoint_regressors"):
        model.freeze_keypoint_regressors()
        print("Keypoint regressors frozen (no grad / weight decay).")

    # Create data loaders (CSV-only pipeline)
    if not args.data_root:
        raise ValueError("--data-root is required for CSV training data")

    persistent_workers = args.num_workers > 0

    train_loader = create_dataloader(
        csv_path=args.train_data,
        root_dir=args.data_root,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        target_size=target_size,
        augment=True,
        persistent_workers=persistent_workers
    )
    print(f'Training samples: {len(train_loader.dataset)}')

    val_loader = None
    if args.val_data:
        val_loader = create_dataloader(
            csv_path=args.val_data,
            root_dir=args.data_root,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            target_size=target_size,
            augment=False,
            persistent_workers=persistent_workers
        )
        print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Create loss function
    loss_fn = BlazeFaceDetectionLoss(
        hard_negative_ratio=args.hard_negative_ratio,
        detection_weight=args.detection_weight,
        classification_weight=args.classification_weight,
        scale=scale,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma
    )
    
    # Create optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create trainer
    trainer = BlazeFaceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name='BlazeFace',
        scale=scale
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f'\nFound checkpoint: {checkpoint_path}')
            print('Resuming training from checkpoint...')
            trainer.load_checkpoint(str(checkpoint_path))
        else:
            print(f'Warning: specified checkpoint {checkpoint_path} not found. Starting fresh.')

    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
