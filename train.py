"""
Training script for ear detection and landmark models.

Supports training BlazeFace-style models with configurable:
- Model architecture
- Data loading
- Loss functions
- Optimization settings
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader, DetectorDataset, LandmarkerDataset
from loss_functions import get_loss, BlazeFaceDetectionLoss


class Trainer:
    """
    Generic trainer for ear detection/landmark models.
    
    Handles training loop, validation, checkpointing, and logging.
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
        model_name: str = 'model'
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            loss_fn: Loss function (auto-detected if None)
            optimizer: Optimizer (AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
            model_name: Name for saving checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Setup loss function
        self.loss_fn = loss_fn if loss_fn else self._auto_detect_loss()
        
        # Setup optimizer
        self.optimizer = optimizer if optimizer else optim.AdamW(
            model.parameters(),
            lr=1e-3,
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
    
    def _auto_detect_loss(self) -> nn.Module:
        """Auto-detect appropriate loss function based on model."""
        # Check model attributes to determine type
        if hasattr(self.model, 'num_anchors'):
            return get_loss('detection')
        elif hasattr(self.model, 'num_keypoints') and not hasattr(self.model, 'num_anchors'):
            return get_loss('landmark')
        else:
            # Default to detection loss
            return get_loss('detection')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        epoch_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self._forward_pass(batch)
            losses = self._compute_loss(outputs, batch)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item()
            
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value.item(), self.global_step)
            
            # Print progress
            if batch_idx % 50 == 0:
                loss_str = ', '.join(f'{k}: {v.item():.4f}' for k, v in losses.items())
                print(f'  Batch {batch_idx}/{len(self.train_loader)} - {loss_str}')
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of average validation losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        val_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                outputs = self._forward_pass(batch)
                losses = self._compute_loss(outputs, batch)
                
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
            self.writer.add_scalar(f'val/{key}', val_losses[key], self.global_step)
        
        return val_losses
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run forward pass through model.
        
        Override this method for custom model architectures.
        
        Args:
            batch: Input batch
            
        Returns:
            Model outputs
        """
        images = batch['image']
        
        # Default forward pass - assumes model returns (scores, coords) or similar
        output = self.model(images)
        
        # Handle different output formats
        if isinstance(output, tuple):
            if len(output) == 2:
                return {'scores': output[0], 'coords': output[1]}
            elif len(output) == 3:
                return {'scores': output[0], 'coords': output[1], 'keypoints': output[2]}
        elif isinstance(output, dict):
            return output
        else:
            # Single output (e.g., keypoints only)
            return {'keypoints': output}
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Override this method for custom loss computation.
        
        Args:
            outputs: Model outputs
            batch: Input batch with targets
            
        Returns:
            Dictionary of losses
        """
        # Build targets dict
        targets = {}
        
        if 'bboxes' in batch:
            # For detection: need to encode targets to anchor format
            # This is a simplified version - you may need custom encoding
            targets['coords'] = batch['bboxes']
            targets['scores'] = batch.get('bbox_mask', 
                torch.ones(batch['bboxes'].shape[:2], device=self.device))
        
        if 'keypoints' in batch:
            targets['keypoints'] = batch['keypoints']
        
        if 'visibility' in batch:
            targets['visibility'] = batch['visibility']
        
        # Compute loss based on available outputs
        if 'scores' in outputs and 'coords' in outputs:
            # Detection loss
            return self.loss_fn(
                outputs['scores'],
                outputs['coords'],
                targets.get('scores', torch.zeros_like(outputs['scores'])),
                targets.get('coords', torch.zeros_like(outputs['coords']))
            )
        elif 'keypoints' in outputs:
            # Landmark loss
            return self.loss_fn(
                outputs['keypoints'],
                targets['keypoints'],
                targets.get('visibility')
            )
        else:
            raise ValueError("Unknown output format")
    
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
        print(f'Saved checkpoint: {path}')
        
        if is_best:
            best_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f'Saved best model: {best_path}')
    
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
        save_every: int = 5,
        validate_every: int = 1
    ):
        """
        Run full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            validate_every: Run validation every N epochs
        """
        print(f'Starting training for {num_epochs} epochs')
        print(f'Device: {self.device}')
        print(f'Model: {self.model_name}')
        print(f'Training samples: {len(self.train_loader.dataset)}')
        if self.val_loader:
            print(f'Validation samples: {len(self.val_loader.dataset)}')
        print('-' * 50)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f'\nEpoch {epoch + 1}/{start_epoch + num_epochs}')
            
            # Train
            train_losses = self.train_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/lr', current_lr, self.global_step)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            loss_str = ', '.join(f'{k}: {v:.4f}' for k, v in train_losses.items())
            print(f'  Train - {loss_str} ({epoch_time:.1f}s)')
            
            # Validate
            if self.val_loader and (epoch + 1) % validate_every == 0:
                val_losses = self.validate()
                loss_str = ', '.join(f'{k}: {v:.4f}' for k, v in val_losses.items())
                print(f'  Val   - {loss_str}')
                
                # Check for best model
                if val_losses.get('total', float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.save_checkpoint(is_best=True)
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint()
        
        # Save final checkpoint
        self.save_checkpoint(f'{self.model_name}_final.pth')
        self.writer.close()
        
        print('\nTraining complete!')


def train_model(
    model_class: Type[nn.Module],
    train_npy: str,
    val_npy: Optional[str] = None,
    dataset_type: str = 'detector',
    loss_type: str = 'detection',
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cuda',
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    model_kwargs: Optional[Dict] = None,
    resume_from: Optional[str] = None
):
    """
    High-level training function.
    
    Args:
        model_class: Model class to instantiate
        train_npy: Path to training NPY file
        val_npy: Path to validation NPY file
        dataset_type: Type of dataset ('detector', 'landmarker', 'teacher')
        loss_type: Type of loss ('detection', 'landmark', 'combined')
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device ('cuda' or 'cpu')
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
        model_kwargs: Additional arguments for model initialization
        resume_from: Path to checkpoint to resume from
    """
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    
    # Create model
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    model_name = model_class.__name__
    
    print(f'Created model: {model_name}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Create data loaders
    train_loader = get_dataloader(
        dataset_type=dataset_type,
        npy_path=train_npy,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = None
    if val_npy:
        val_loader = get_dataloader(
            dataset_type=dataset_type,
            npy_path=val_npy,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
    
    # Create loss function
    loss_fn = get_loss(loss_type)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        model_name=model_name
    )
    
    # Resume if specified
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return trainer


def main():
    """Main entry point for command-line training."""
    parser = argparse.ArgumentParser(description='Train ear detection/landmark models')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model class name (e.g., BlazeFace, BlazeFaceLandmark)')
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training NPY file')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Path to validation NPY file')
    parser.add_argument('--dataset-type', type=str, default='detector',
                        choices=['detector', 'landmarker', 'teacher'],
                        help='Type of dataset')
    parser.add_argument('--loss-type', type=str, default='detection',
                        choices=['detection', 'landmark', 'combined'],
                        help='Type of loss function')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Import model class dynamically
    if args.model == 'BlazeFace':
        from blazeface import BlazeFace as ModelClass
    elif args.model == 'BlazeFaceLandmark':
        from blazeface_landmark import BlazeFaceLandmark as ModelClass
    else:
        # Try to import from blazeface module
        try:
            import importlib
            module = importlib.import_module('blazeface')
            ModelClass = getattr(module, args.model)
        except (ImportError, AttributeError):
            raise ValueError(f"Unknown model: {args.model}")
    
    # Train
    train_model(
        model_class=ModelClass,
        train_npy=args.train_data,
        val_npy=args.val_data,
        dataset_type=args.dataset_type,
        loss_type=args.loss_type,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
