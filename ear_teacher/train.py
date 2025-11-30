"""Training script for Ear VAE with PyTorch Lightning."""

import argparse
from pathlib import Path
import sys
import warnings

# Suppress pydantic serialization warnings from albumentations
warnings.filterwarnings('ignore', message='.*Pydantic serializer warnings.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic.main')

# Suppress ImageCompression float32 warnings from albumentations
warnings.filterwarnings('ignore', message='.*Image compression augmentation is most effective with uint8 inputs.*')
warnings.filterwarnings('ignore', category=UserWarning, module='albumentations.augmentations.functional')

# Suppress NaN warnings from torchmetrics
warnings.filterwarnings('ignore', message='.*Encounted `nan` values in tensor.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics.aggregation')

# Support both standalone and module execution
if __name__ == '__main__':
    # Add parent directory to path for standalone execution
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import CSVLogger

# Handle both relative and absolute imports
try:
    from .lightning_module import EarVAELightning
    from .datamodule import EarDataModule
except ImportError:
    from ear_teacher.lightning_module import EarVAELightning
    from ear_teacher.datamodule import EarDataModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Ear VAE model with PyTorch Lightning')

    # Data
    parser.add_argument('--train_data', type=str, default='data/preprocessed/train_teacher.npy',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/preprocessed/val_teacher.npy',
                       help='Path to validation data')

    # Model
    parser.add_argument('--in_channels', type=int, default=3,
                       help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension size')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--kld_weight', type=float, default=0.025,
                       help='KL divergence weight')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                       help='Perceptual loss weight (0.0 = disabled)')
    parser.add_argument('--ssim_weight', type=float, default=0.0,
                       help='SSIM loss weight (0.0 = disabled)')
    parser.add_argument('--edge_weight', type=float, default=0.0,
                       help='Edge loss weight (0.0 = disabled)')

    # KLD Annealing
    parser.add_argument('--kld_anneal_strategy', type=str, default='cyclic',
                       choices=['linear', 'cyclic', 'monotonic'],
                       help='KLD annealing strategy (linear, cyclic, monotonic)')
    parser.add_argument('--kld_anneal_cycles', type=int, default=4,
                       help='Number of cycles for cyclic annealing')
    parser.add_argument('--kld_anneal_ratio', type=float, default=0.5,
                       help='Ratio of increasing phase in each cycle (0.0-1.0)')
    parser.add_argument('--kld_anneal_start', type=float, default=0.0,
                       help='Starting weight for KLD annealing')
    parser.add_argument('--kld_anneal_end', type=float, default=1.0,
                       help='Ending weight multiplier for KLD annealing')

    # Logging and checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='ear_teacher/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='ear_teacher/logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--experiment_name', type=str, default='ear_vae',
                       help='Experiment name for logging')

    # Hardware
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Accelerator to use (auto, gpu, cpu, tpu)')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices to use')
    parser.add_argument('--precision', type=str, default='32',
                       help='Precision (32, 16, bf16)')

    # Other
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run a quick test with 1 batch')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DataModule
    datamodule = EarDataModule(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Print dataset info
    print("\n" + "="*80)
    print("Dataset Information")
    print("="*80)
    num_samples = datamodule.get_num_samples()
    print(f"Training samples: {num_samples['train']:,}")
    print(f"Validation samples: {num_samples['val']:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training batches per epoch: {num_samples['train'] // args.batch_size}")
    print("="*80 + "\n")

    # Initialize model
    model = EarVAELightning(
        in_channels=args.in_channels,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kld_weight=args.kld_weight,
        kld_anneal_strategy=args.kld_anneal_strategy,
        kld_anneal_cycles=args.kld_anneal_cycles,
        kld_anneal_ratio=args.kld_anneal_ratio,
        kld_anneal_start=args.kld_anneal_start,
        kld_anneal_end=args.kld_anneal_end,
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
    )

    # Print model info
    print("Model Information")
    print("="*80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*80 + "\n")

    # Callbacks
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:03d}-{val/loss:.6f}',
            monitor='val/loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
        ),
        # Save best model based on SSIM
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-ssim-{epoch:03d}-{val/ssim:.4f}',
            monitor='val/ssim',
            mode='max',
            save_top_k=1,
            verbose=False,
        ),
        # Early stopping
        EarlyStopping(
            monitor='val/loss',
            patience=30,
            mode='min',
            verbose=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='epoch'),
        # Rich progress bar
        RichProgressBar(),
    ]

    # Logger
    logger = CSVLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate after each epoch
        gradient_clip_val=1.0,
        deterministic=False,  # Set to True for full reproducibility (slower)
        fast_dev_run=args.fast_dev_run,
        enable_model_summary=True,
    )

    # Find best checkpoint if exists and no explicit resume path provided
    if args.resume is None:
        best_checkpoint = checkpoint_dir / 'last.ckpt'
        if best_checkpoint.exists():
            args.resume = str(best_checkpoint)
            print(f"Resuming training from: {args.resume}\n")
        else:
            print("Starting training from scratch\n")
    else:
        print(f"Resuming training from: {args.resume}\n")

    # Train
    print("Starting training...")
    print("="*80 + "\n")

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume if args.resume and Path(args.resume).exists() else None
    )

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best model saved at: {checkpoint_dir}")
    print(f"Training logs at: {logger.log_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
