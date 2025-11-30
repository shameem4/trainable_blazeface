"""
Training script for Ear Teacher VAE.
"""

import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import warnings

from lightning import EarVAELightning, EarTeacherDataModule


def main():
    # Suppress Pydantic serialization warnings from albumentations
    warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
    # Suppress scheduler epoch parameter deprecation warning
    warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')
    # Suppress model summary precision warning
    warnings.filterwarnings('ignore', message='.*Precision.*is not supported by the model summary.*')
    # Suppress pkg_resources deprecation warning from torchmetrics
    warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
    warnings.filterwarnings('ignore', category=DeprecationWarning, module='pkg_resources')
    warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics.utilities.imports')
    # Suppress frozen modules warning from PyTorch Lightning (expected for frozen DINOv2 backbone)
    warnings.filterwarnings('ignore', message='.*Found.*module.*in eval mode at the start of training.*')

    # Set matrix multiplication precision for better performance on GPUs with Tensor Cores
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description='Train Ear Teacher VAE')

    # Data arguments
    parser.add_argument('--train-npy', type=str, default='data/preprocessed/train_teacher.npy',
                        help='Path to training NPY file')
    parser.add_argument('--val-npy', type=str, default='data/preprocessed/val_teacher.npy',
                        help='Path to validation NPY file')
    parser.add_argument('--root-dir', type=str, default=None,
                        help='Root directory for image paths')

    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=1024,
                        help='Latent space dimensionality (higher = more detail capacity)')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Input image size (square)')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (reduced: faster convergence expected)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (for custom layers, DINOv2 uses 0.1x this)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='Learning rate scheduler')

    # Loss weights (Option 2: Optimized for sharp reconstructions)
    parser.add_argument('--kl-weight', type=float, default=0.000001,
                        help='KL weight (ultra-low for maximum detail preservation)')
    parser.add_argument('--perceptual-weight', type=float, default=1.5,
                        help='Perceptual loss (stronger for sharp semantic features)')
    parser.add_argument('--ssim-weight', type=float, default=0.6,
                        help='SSIM loss (stronger structural preservation)')
    parser.add_argument('--edge-weight', type=float, default=0.3,
                        help='Edge/gradient loss (strong emphasis on sharp boundaries)')
    parser.add_argument('--contrastive-weight', type=float, default=0.1,
                        help='Contrastive loss weight (feature discrimination)')
    parser.add_argument('--center-weight', type=float, default=3.0,
                        help='Center region weight (higher = more focus on ear center)')
    parser.add_argument('--recon-loss', type=str, default='l1',
                        choices=['mse', 'l1'],
                        help='Reconstruction loss type (L1 better for sharp details)')

    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Training precision (32, 16-mixed, bf16-mixed)')

    # Checkpoint and logging (relative to ear_teacher directory)
    parser.add_argument('--save-dir', type=str, default='ear_teacher/checkpoints',
                        help='Directory to save checkpoints (within ear_teacher)')
    parser.add_argument('--log-dir', type=str, default='ear_teacher/logs',
                        help='Directory for tensorboard logs (within ear_teacher)')
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

    # Model (DINOv2 hybrid encoder is now default)
    model = EarVAELightning(
        latent_dim=args.latent_dim,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        center_weight=args.center_weight,
        contrastive_weight=args.contrastive_weight,
        recon_loss_type=args.recon_loss,
        warmup_epochs=args.warmup_epochs,
        scheduler=args.scheduler,
        image_size=args.image_size
    )

    print("\nUsing DINOv2 hybrid encoder (pretrained DINOv2 backbone)\n")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='ear_vae-{epoch:03d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )

    callbacks = [
        checkpoint_callback,
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
    logger = CSVLogger(
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
        accumulate_grad_batches=2,  # Effective batch size = 32 * 2 = 64
        deterministic=False,
        benchmark=True,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume
    )

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Copy best model to production location
    import shutil
    prod_dir = Path('models/ear_teacher')
    prod_dir.mkdir(parents=True, exist_ok=True)
    prod_path = prod_dir / 'ear_teacher.ckpt'

    if checkpoint_callback.best_model_path:
        shutil.copy2(checkpoint_callback.best_model_path, prod_path)
        print(f"Best model copied to: {prod_path}")

    print(f"CSV logs: {args.log_dir}")


if __name__ == '__main__':
    main()
