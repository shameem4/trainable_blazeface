"""
Training script for BlazeFaceLandmark ear landmark model.

Usage:
    python train_blazeface_landmark.py --train-data data/preprocessed/train_landmarker.npy
    python train_blazeface_landmark.py --train-data data/preprocessed/train_landmarker.npy --val-data data/preprocessed/val_landmarker.npy
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from blazeface_landmark import BlazeFaceLandmark
from dataloader import get_dataloader, LandmarkerDataset
from loss import LandmarkLoss
from train import Trainer


class BlazeFaceLandmarkTrainer(Trainer):
    """
    Specialized trainer for BlazeFaceLandmark model.
    
    Handles the specific output format and loss computation for landmark prediction.
    """
    
    def _forward_pass(self, batch):
        """
        Forward pass for BlazeFaceLandmark.
        
        BlazeFaceLandmark outputs (flag, landmarks) where:
        - flag: confidence score for face presence
        - landmarks: [B, 468, 3] normalized landmark coordinates
        """
        images = batch['image']
        
        # BlazeFaceLandmark forward returns (flag, landmarks)
        flag, landmarks = self.model(images)
        
        return {
            'flag': flag,
            'keypoints': landmarks  # [B, 468, 3] -> x, y, z normalized
        }
    
    def _compute_loss(self, outputs, batch):
        """
        Compute landmark loss.
        
        Args:
            outputs: Dict with 'flag' and 'keypoints'
            batch: Dict with ground truth keypoints
            
        Returns:
            Loss dictionary
        """
        pred_keypoints = outputs['keypoints']  # [B, 468, 3]
        pred_flag = outputs['flag']  # [B]
        
        # Get target keypoints
        target_keypoints = batch['keypoints']  # [B, K, 2] or [B, K*2]
        
        # Reshape if needed to match prediction format
        batch_size = pred_keypoints.shape[0]
        num_pred_kpts = pred_keypoints.shape[1]
        
        # Handle dimension mismatch - target may have different number of keypoints
        if target_keypoints.dim() == 2:
            # [B, K*2] -> [B, K, 2]
            target_keypoints = target_keypoints.view(batch_size, -1, 2)
        
        num_target_kpts = target_keypoints.shape[1]
        
        # If target has fewer keypoints, pad or use subset of predictions
        if num_target_kpts < num_pred_kpts:
            # Use only the first N predicted keypoints
            pred_for_loss = pred_keypoints[:, :num_target_kpts, :2]
        elif num_target_kpts > num_pred_kpts:
            # Use only the first N target keypoints
            target_keypoints = target_keypoints[:, :num_pred_kpts, :]
            pred_for_loss = pred_keypoints[:, :, :2]
        else:
            pred_for_loss = pred_keypoints[:, :, :2]
        
        # Compute landmark loss
        landmark_losses = self.loss_fn(
            pred_for_loss,
            target_keypoints,
            batch.get('visibility')
        )
        
        # Add flag loss if we have face presence labels
        if 'flag' in batch:
            target_flag = batch['flag']
            flag_loss = nn.functional.binary_cross_entropy(
                pred_flag, target_flag
            )
            landmark_losses['flag'] = flag_loss
            landmark_losses['total'] = landmark_losses['total'] + 0.5 * flag_loss
        
        return landmark_losses


def create_landmark_model(pretrained: bool = False) -> BlazeFaceLandmark:
    """
    Create BlazeFaceLandmark model.
    
    Args:
        pretrained: Whether to load pretrained weights
        
    Returns:
        BlazeFaceLandmark model
    """
    model = BlazeFaceLandmark()
    
    if pretrained:
        weights_path = Path('model_weights/blazeface_landmark.pth')
        if weights_path.exists():
            print(f'Loading pretrained weights from {weights_path}')
            state_dict = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            print(f'Warning: Pretrained weights not found at {weights_path}')
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train BlazeFaceLandmark ear landmark model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training NPY file')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation NPY file')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory for image paths')
    
    # Model arguments
    parser.add_argument('--pretrained', action='store_true',
                        help='Load pretrained weights for fine-tuning')
    parser.add_argument('--num-keypoints', type=int, default=55,
                        help='Number of keypoints to predict')
    
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
    parser.add_argument('--loss-type', type=str, default='wing',
                        choices=['l1', 'l2', 'smooth_l1', 'wing'],
                        help='Type of landmark loss')
    parser.add_argument('--wing-w', type=float, default=10.0,
                        help='Wing loss width parameter')
    parser.add_argument('--wing-epsilon', type=float, default=2.0,
                        help='Wing loss epsilon parameter')
    
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
    
    # BlazeFaceLandmark uses 192x192 input
    target_size = (192, 192)
    
    print('=' * 60)
    print('BlazeFaceLandmark Ear Landmark Training')
    print('=' * 60)
    print(f'Input size: {target_size}')
    print(f'Number of keypoints: {args.num_keypoints}')
    print(f'Device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Learning rate: {args.lr}')
    print(f'Loss type: {args.loss_type}')
    print(f'Epochs: {args.epochs}')
    print('=' * 60)
    
    # Create model
    model = create_landmark_model(pretrained=args.pretrained)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create data loaders
    train_loader = get_dataloader(
        dataset_type='landmarker',
        npy_path=args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        root_dir=args.root_dir,
        target_size=target_size,
        num_keypoints=args.num_keypoints
    )
    print(f'Training samples: {len(train_loader.dataset)}')
    
    val_loader = None
    if args.val_data:
        val_loader = get_dataloader(
            dataset_type='landmarker',
            npy_path=args.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            root_dir=args.root_dir,
            target_size=target_size,
            num_keypoints=args.num_keypoints
        )
        print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Create loss function
    loss_fn = LandmarkLoss(
        num_keypoints=args.num_keypoints,
        loss_type=args.loss_type,
        wing_w=args.wing_w,
        wing_epsilon=args.wing_epsilon
    )
    
    # Create optimizer with different learning rates for backbone and head
    # Lower LR for pretrained backbone, higher for head
    if args.pretrained:
        backbone_params = list(model.backbone1.parameters())
        head_params = list(model.backbone2a.parameters()) + list(model.backbone2b.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': head_params, 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=args.lr * 0.001
    )
    
    # Create trainer
    trainer = BlazeFaceLandmarkTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name='BlazeFaceLandmark'
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
    print(f'Best model saved to: {args.checkpoint_dir}/BlazeFaceLandmark_best.pth')


if __name__ == '__main__':
    main()
