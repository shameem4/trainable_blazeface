"""
Training script for BlazeFace ear detector.

Usage:
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy
    python train_blazeface.py --train-data data/preprocessed/train_detector.npy --val-data data/preprocessed/val_detector.npy
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from blazeface import BlazeFace
from dataloader import get_dataloader, DetectorDataset
from loss_functions import BlazeFaceDetectionLoss
from train import Trainer


class BlazeFaceTrainer(Trainer):
    """
    Specialized trainer for BlazeFace detector.
    
    Handles the specific output format and loss computation for BlazeFace.
    """
    
    def _forward_pass(self, batch):
        """
        Forward pass for BlazeFace.
        
        BlazeFace outputs raw classification and regression tensors
        that need to be processed for training.
        """
        images = batch['image']
        
        # BlazeFace forward returns (raw_scores, raw_coords) or processed detections
        # For training, we need raw outputs before NMS
        if hasattr(self.model, 'forward_train'):
            scores, coords = self.model.forward_train(images)
        else:
            # Use the backbone directly to get raw predictions
            scores, coords = self._get_raw_predictions(images)
        
        return {
            'scores': scores,
            'coords': coords
        }
    
    def _get_raw_predictions(self, images):
        """Get raw predictions from BlazeFace backbone."""
        # This accesses the internal structure of BlazeFace
        # to get predictions before post-processing
        x = self.model._preprocess(images)
        
        if self.model.back_model:
            x = self.model.backbone(x)
            c1 = self.model.classifier_8(x)
            c2 = self.model.classifier_16(x)
            c = torch.cat([c1.view(-1, 2, 16*16), c2.view(-1, 6, 8*8)], dim=-1)
            
            r1 = self.model.regressor_8(x)
            r2 = self.model.regressor_16(x)
            r = torch.cat([r1.view(-1, 16, 16*16), r2.view(-1, 16, 8*8)], dim=-1)
        else:
            x = self.model.backbone1(x)
            h = self.model.backbone2(x)
            
            c = self.model.classifier(x)
            r = self.model.regressor(x)
            
            c = c.permute(0, 2, 3, 1).reshape(-1, self.model.num_anchors, self.model.num_classes)
            r = r.permute(0, 2, 3, 1).reshape(-1, self.model.num_anchors, self.model.num_coords)
        
        return c, r
    
    def _compute_loss(self, outputs, batch):
        """
        Compute detection loss for BlazeFace.
        
        Args:
            outputs: Dict with 'scores' and 'coords'
            batch: Dict with ground truth bboxes
            
        Returns:
            Loss dictionary
        """
        pred_scores = outputs['scores']
        pred_coords = outputs['coords']
        
        # For now, create dummy targets - you'll need to implement
        # proper anchor-based target encoding
        batch_size = pred_scores.shape[0]
        num_anchors = pred_scores.shape[1]
        
        # Create target tensors (zeros = no detection)
        target_scores = torch.zeros_like(pred_scores)
        target_coords = torch.zeros_like(pred_coords)
        
        # TODO: Implement proper anchor matching
        # This requires matching ground truth boxes to anchors
        # and encoding the targets appropriately
        if 'bboxes' in batch and batch['bboxes'].numel() > 0:
            # Simplified: mark first anchor as positive for each bbox
            # Real implementation needs IoU-based matching
            pass
        
        return self.loss_fn(
            pred_scores,
            pred_coords,
            target_scores,
            target_coords
        )


def create_blazeface_model(back_model: bool = False, pretrained: bool = False) -> BlazeFace:
    """
    Create BlazeFace model.
    
    Args:
        back_model: Whether to use back-facing camera model (256x256) vs front (128x128)
        pretrained: Whether to load pretrained weights
        
    Returns:
        BlazeFace model
    """
    model = BlazeFace(back_model=back_model)
    
    if pretrained:
        weights_path = Path('model_weights/blazeface.pth')
        if weights_path.exists():
            print(f'Loading pretrained weights from {weights_path}')
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f'Warning: Pretrained weights not found at {weights_path}')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train BlazeFace ear detector')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training NPY file')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation NPY file')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory for image paths')
    
    # Model arguments
    parser.add_argument('--back-model', action='store_true',
                        help='Use back-facing camera model (256x256 input)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained weights')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Loss arguments
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma')
    parser.add_argument('--reg-weight', type=float, default=1.0,
                        help='Regression loss weight')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        args.device = 'cpu'
    
    # Determine input size based on model type
    target_size = (256, 256) if args.back_model else (128, 128)
    
    print('=' * 60)
    print('BlazeFace Ear Detector Training')
    print('=' * 60)
    print(f'Model type: {"back" if args.back_model else "front"}')
    print(f'Input size: {target_size}')
    print(f'Device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Epochs: {args.epochs}')
    print('=' * 60)
    
    # Create model
    model = create_blazeface_model(
        back_model=args.back_model,
        pretrained=args.pretrained
    )
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create data loaders
    train_loader = get_dataloader(
        dataset_type='detector',
        npy_path=args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        root_dir=args.root_dir,
        target_size=target_size,
        num_anchors=model.num_anchors,
        num_keypoints=model.num_keypoints
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    
    val_loader = None
    if args.val_data:
        val_loader = get_dataloader(
            dataset_type='detector',
            npy_path=args.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            root_dir=args.root_dir,
            target_size=target_size,
            num_anchors=model.num_anchors,
            num_keypoints=model.num_keypoints
        )
        print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Create loss function
    loss_fn = DetectionLoss(
        num_classes=model.num_classes,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        regression_weight=args.reg_weight,
        use_focal_loss=True
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
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
        model_name='BlazeFace'
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every
    )
    
    print('\nTraining complete!')
    print(f'Best model saved to: {args.checkpoint_dir}/BlazeFace_best.pth')


if __name__ == '__main__':
    main()
