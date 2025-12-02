"""
Training script for ear teacher model.

Run standalone from root directory:
    python ear_teacher/train.py
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# Add parent directory to path for standalone execution
script_dir = Path(__file__).parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from ear_teacher.datamodule import EarDataModule
from ear_teacher.lightning_module import EarTeacherLightningModule


def main():
    """Main training function."""
    # Parse arguments
    parser = ArgumentParser(description="Train ear teacher model")

    # Data arguments
    parser.add_argument(
        "--train_metadata",
        type=str,
        default="common/data/preprocessed/train_teacher.npy",
        help="Path to training metadata",
    )
    parser.add_argument(
        "--val_metadata",
        type=str,
        default="common/data/preprocessed/val_teacher.npy",
        help="Path to validation metadata",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory for image paths",
    )
    parser.add_argument(
        "--bbox_buffer",
        type=float,
        default=0.10,
        help="Bbox buffer percentage (default: 0.10 for 10%%)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable augmentations (disabled by default, ramp up later)",
    )

    # Model arguments
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="models/convnext_tiny_22k_224.pth",
        help="Path to pretrained ConvNeXt weights",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=768,
        help="ConvNeXt-Tiny embedding dimension",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=256,
        help="Projection head output dimension",
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    parser.add_argument(
        "--metric_weight",
        type=float,
        default=0.1,
        help="Weight for metric learning loss (ArcFace/CosFace)",
    )
    parser.add_argument(
        "--metric_loss",
        type=str,
        default="arcface",
        choices=["arcface", "cosface", "none"],
        help="Type of metric learning loss",
    )
    parser.add_argument(
        "--num_pseudo_classes",
        type=int,
        default=512,
        help="Number of pseudo-classes for metric learning",
    )
    parser.add_argument(
        "--arcface_margin",
        type=float,
        default=0.5,
        help="Angular margin for ArcFace (or margin for CosFace)",
    )
    parser.add_argument(
        "--arcface_scale",
        type=float,
        default=64.0,
        help="Feature scale for ArcFace/CosFace",
    )
    parser.add_argument(
        "--num_collage_samples",
        type=int,
        default=10,
        help="Number of samples for validation collage",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator type (auto, gpu, cpu)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Training precision (32, 16-mixed, bf16-mixed)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ear_teacher/outputs/",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="ear_teacher",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data module
    datamodule = EarDataModule(
        train_metadata_path=args.train_metadata,
        val_metadata_path=args.val_metadata,
        root_dir=args.root_dir,
        bbox_buffer=args.bbox_buffer,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )

    # Initialize model
    model = EarTeacherLightningModule(
        pretrained_path=args.pretrained_path,
        embedding_dim=args.embedding_dim,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        reconstruction_weight=args.reconstruction_weight,
        metric_weight=args.metric_weight,
        num_collage_samples=args.num_collage_samples,
        metric_loss=args.metric_loss,
        num_pseudo_classes=args.num_pseudo_classes,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
    )

    # Setup logger
    logger = CSVLogger(
        save_dir=args.output_dir,
        name=args.experiment_name,
    )

    # Setup callbacks
    callbacks = [
        # Model checkpoint - save best model based on validation loss
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.experiment_name, "checkpoints"),
            filename="epoch={epoch:03d}-val_loss={val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        # Periodic checkpoint
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, args.experiment_name, "checkpoints"),
            filename="epoch={epoch:03d}",
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_top_k=0,  # Don't limit periodic checkpoints (every_n_epochs controls saving)
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
        # Rich progress bar
        RichProgressBar(),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    # Print configuration
    print("\n" + "=" * 80)
    print("EAR TEACHER TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"\nData:")
    print(f"  Train metadata: {args.train_metadata}")
    print(f"  Val metadata: {args.val_metadata}")
    print(f"  Image size: {args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Bbox buffer: {args.bbox_buffer * 100:.1f}%")
    print(f"  Augmentation: {args.augment}")
    print(f"\nModel:")
    print(f"  Pretrained path: {args.pretrained_path}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Projection dim: {args.projection_dim}")
    print(f"\nTraining:")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Reconstruction weight: {args.reconstruction_weight}")
    print(f"  Metric weight: {args.metric_weight}")
    print(f"  Metric loss type: {args.metric_loss}")
    if args.metric_loss != "none":
        print(f"  Pseudo-classes: {args.num_pseudo_classes}")
        print(f"  Margin: {args.arcface_margin}")
        print(f"  Scale: {args.arcface_scale}")
    print(f"  Precision: {args.precision}")
    print(f"  Collage samples: {args.num_collage_samples}")
    print(f"\nOutput:")
    print(f"  Directory: {args.output_dir}")
    print(f"  Experiment: {args.experiment_name}")
    print("=" * 80 + "\n")

    # Train
    trainer.fit(model, datamodule=datamodule)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Logs saved to: {logger.log_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
