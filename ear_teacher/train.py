"""
Training script for Ear Teacher VAE.
"""

import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lightning import EarVAELightning, EarTeacherDataModule


def main():
    parser = argparse.ArgumentParser(description='Train Ear Teacher VAE')

    # Data arguments
    parser.add_argument('--train-npy', type=str, default='data/preprocessed/train_teacher.npy',
                        help='Path to training NPY file')
    parser.add_argument('--val-npy', type=str, default='data/preprocessed/val_teacher.npy',
                        help='Path to validation NPY file')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory for image paths')

    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=512,
                        help='Latent space dimensionality')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Input image size (square)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')

    # Loss weights
    parser.add_argument('--kl-weight', type=float, default=0.0001,
                        help='KL divergence weight')
    parser.add_argument('--perceptual-weight', type=float, default=0.5,
                        help='Perceptual loss weight')
    parser.add_argument('--ssim-weight', type=float, default=0.1,
                        help='SSIM loss weight')
    parser.add_argument('--recon-loss', type=str, default='mse',
                        choices=['mse', 'l1'],
                        help='Reconstruction loss type')

    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Training precision (32, 16-mixed, bf16-mixed)')

    # Checkpoint and logging
    parser.add_argument('--save-dir', type=str, default='checkpoints/ear_teacher',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/ear_teacher',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Early stopping
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # DataModule
    datamodule = EarTeacherDataModule(
        train_npy=args.train_npy,
        val_npy=args.val_npy,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        root_dir=args.root_dir
    )

    # Model
    model = EarVAELightning(
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        recon_loss_type=args.recon_loss,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename='ear_vae-{epoch:03d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val/loss',
                patience=args.patience,
                mode='min',
                verbose=True
            )
        )

    # Logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='ear_vae'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=False,
        benchmark=True
    )

    # Train
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume
    )

    print(f"\nTraining complete!")
    print(f"Best model checkpoint: {callbacks[0].best_model_path}")
    print(f"TensorBoard logs: {args.log_dir}")


if __name__ == '__main__':
    main()
