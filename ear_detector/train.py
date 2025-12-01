"""
Training script for BlazeEar detector.

Run standalone from root directory:
    python ear_detector/train.py

Metrics tracked (matching BlazeFace paper):
- mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
- Precision, Recall
- Inference speed (FPS)
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

from ear_detector.datamodule import EarDetectorDataModule
from ear_detector.lightning_module import BlazeEarLightningModule


def main():
    """Main training function."""
    parser = ArgumentParser(description="Train BlazeEar detector")
    
    # Data arguments
    parser.add_argument(
        "--train_metadata",
        type=str,
        default="data/preprocessed/train_detector.npy",
        help="Path to training metadata",
    )
    parser.add_argument(
        "--val_metadata",
        type=str,
        default="data/preprocessed/val_detector.npy",
        help="Path to validation metadata",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Root directory for image paths",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Input image size (128 for BlazeFace)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable augmentations",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable augmentations",
    )
    
    # Model arguments
    parser.add_argument(
        "--pretrained_blazeface",
        type=str,
        default="mediapipe/BlazeFace/blazeface.pth",
        help="Path to pretrained BlazeFace weights (default: mediapipe weights)",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train from scratch without pretrained weights",
    )
    parser.add_argument(
        "--num_anchors_16",
        type=int,
        default=2,
        help="Number of anchors at 16x16 scale",
    )
    parser.add_argument(
        "--num_anchors_8",
        type=int,
        default=6,
        help="Number of anchors at 8x8 scale",
    )
    parser.add_argument(
        "--anchor_config",
        type=str,
        default="data/preprocessed/detector_anchors.npy",
        help="Path to anchor config file (from create_detector_anchors.py)",
    )
    
    # Loss arguments
    parser.add_argument(
        "--pos_iou_threshold",
        type=float,
        default=0.35,
        help="IoU threshold for positive anchors (0.35 based on anchor-GT analysis)",
    )
    parser.add_argument(
        "--neg_iou_threshold",
        type=float,
        default=0.2,
        help="IoU threshold for negative anchors (0.2 based on P25 of best IoU)",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma",
    )
    parser.add_argument(
        "--box_weight",
        type=float,
        default=50.0,
        help="Weight for box regression loss (50x to balance with cls loss)",
    )
    
    # Training arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate (heads LR; backbone uses 10x lower)",
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
        default=5,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Epochs to freeze backbone (train heads only). Disabled by default as differential LR works better.",
    )
    
    # Inference arguments
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Score threshold for detection (lower for better recall)",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.4,
        help="NMS IoU threshold (0.4 for ear detection - allows some overlap)",
    )
    
    # Trainer arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator (auto, gpu, cpu)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Training precision",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ear_detector/outputs",
        help="Output directory for logs and checkpoints",
    )
    
    args = parser.parse_args()
    
    # Handle augment flag
    augment = args.augment and not args.no_augment
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data module
    datamodule = EarDetectorDataModule(
        train_metadata=args.train_metadata,
        val_metadata=args.val_metadata,
        root_dir=args.root_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=augment,
    )
    
    # Setup model
    # Handle --no_pretrained flag
    pretrained_path = None if args.no_pretrained else args.pretrained_blazeface
    
    model = BlazeEarLightningModule(
        num_anchors_16=args.num_anchors_16,
        num_anchors_8=args.num_anchors_8,
        input_size=args.image_size,
        pretrained_blazeface_path=pretrained_path,
        anchor_config_path=args.anchor_config,
        pos_iou_threshold=args.pos_iou_threshold,
        neg_iou_threshold=args.neg_iou_threshold,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        box_weight=args.box_weight,
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
            filename="blazeear-{epoch:02d}-{val/mAP_50:.4f}",
            monitor="val/mAP_50",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=output_dir,
        name="blazeear",
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
    )
    
    # Print configuration
    print("\n" + "=" * 60)
    print("BlazeEar Detector Training")
    print("=" * 60)
    print(f"\nData:")
    print(f"  Train metadata: {args.train_metadata}")
    print(f"  Val metadata: {args.val_metadata}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Augmentations: {augment}")
    print(f"\nModel:")
    print(f"  Anchors 16x16: {args.num_anchors_16}")
    print(f"  Anchors 8x8: {args.num_anchors_8}")
    print(f"  Total anchors: {args.num_anchors_16 * 256 + args.num_anchors_8 * 64}")
    print(f"  Anchor config: {args.anchor_config or 'Default (hardcoded)'}")
    print(f"  Pretrained: {pretrained_path or 'None (training from scratch)'}")
    print(f"\nTraining:")
    print(f"  Learning rate (heads): {args.learning_rate}")
    print(f"  Learning rate (backbone): {args.learning_rate * 0.1}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Max epochs: {args.max_epochs}")
    if args.freeze_backbone_epochs > 0:
        print(f"  Freeze backbone: {args.freeze_backbone_epochs} epochs (heads only first)")
    else:
        print(f"  Freeze backbone: disabled (train all from start)")
    print(f"\nMetrics (BlazeFace paper style):")
    print(f"  - mAP@0.5 (primary)")
    print(f"  - mAP@0.75")
    print(f"  - mAP@[0.5:0.95]")
    print(f"  - Precision, Recall")
    print(f"  - FPS (inference speed)")
    print(f"\nOutput: {output_dir}")
    print("=" * 60 + "\n")
    
    # Train
    trainer.fit(model, datamodule)
    
    print("\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
