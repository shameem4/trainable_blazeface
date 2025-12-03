"""
Test script to verify CSV dataloader format conversion and anchor encoding.

This script validates:
1. CSV to MediaPipe format conversion
2. Anchor target encoding
3. Data pipeline integrity
"""

import torch
import numpy as np
from csv_dataloader import CSVDetectorDataset
from pathlib import Path


def test_csv_conversion():
    """Test that CSV data is correctly converted to MediaPipe format."""
    print("Testing CSV to MediaPipe conversion...")

    # Load a sample from the dataset
    dataset = CSVDetectorDataset(
        csv_path="data/splits/train.csv",
        root_dir="data/raw/blazeface",
        target_size=(128, 128),
        augment=False
    )

    # Get first sample
    sample = dataset[0]

    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Anchor targets shape: {sample['anchor_targets'].shape}")
    print(f"Small anchors shape: {sample['small_anchors'].shape}")
    print(f"Big anchors shape: {sample['big_anchors'].shape}")

    # Check anchor targets format
    anchor_targets = sample['anchor_targets']
    print(f"\nAnchor targets dtype: {anchor_targets.dtype}")
    print(f"Anchor targets range: [{anchor_targets.min():.4f}, {anchor_targets.max():.4f}]")

    # Find positive anchors
    positive_mask = anchor_targets[:, 0] > 0.5
    num_positive = positive_mask.sum().item()
    print(f"\nNumber of positive anchors: {num_positive}")

    if num_positive > 0:
        positive_anchors = anchor_targets[positive_mask]
        print(f"Positive anchor targets (first 3):")
        for i in range(min(3, len(positive_anchors))):
            cls, ymin, xmin, ymax, xmax = positive_anchors[i]
            print(f"  [{i}] class={cls:.2f}, ymin={ymin:.4f}, xmin={xmin:.4f}, ymax={ymax:.4f}, xmax={xmax:.4f}")

            # Verify MediaPipe convention
            assert 0 <= ymin <= 1, f"ymin out of range: {ymin}"
            assert 0 <= xmin <= 1, f"xmin out of range: {xmin}"
            assert 0 <= ymax <= 1, f"ymax out of range: {ymax}"
            assert 0 <= xmax <= 1, f"xmax out of range: {xmax}"
            assert ymin < ymax, f"ymin >= ymax: {ymin} >= {ymax}"
            assert xmin < xmax, f"xmin >= xmax: {xmin} >= {xmax}"

            print(f"      width={xmax-xmin:.4f}, height={ymax-ymin:.4f}")

    print("\n[PASS] CSV conversion test passed!")
    return dataset


def test_anchor_encoding():
    """Test that anchor encoding produces valid targets."""
    print("\n\nTesting anchor encoding...")

    dataset = CSVDetectorDataset(
        csv_path="data/splits/train.csv",
        root_dir="data/raw/blazeface",
        target_size=(128, 128),
        augment=False
    )

    # Test multiple samples
    num_samples = min(10, len(dataset))
    total_positives = 0

    for i in range(num_samples):
        sample = dataset[i]
        anchor_targets = sample['anchor_targets']

        # Count positives
        positive_mask = anchor_targets[:, 0] > 0.5
        num_positive = positive_mask.sum().item()
        total_positives += num_positive

        # Verify total anchor count
        assert anchor_targets.shape[0] == 896, f"Expected 896 anchors, got {anchor_targets.shape[0]}"
        assert anchor_targets.shape[1] == 5, f"Expected 5 values per anchor, got {anchor_targets.shape[1]}"

        # Verify class values are 0 or 1
        classes = anchor_targets[:, 0]
        assert torch.all((classes == 0) | (classes == 1)), "Class values must be 0 or 1"

        # Verify box coordinates are in [0, 1]
        boxes = anchor_targets[:, 1:]
        if positive_mask.sum() > 0:
            positive_boxes = boxes[positive_mask]
            assert torch.all(positive_boxes >= 0) and torch.all(positive_boxes <= 1), \
                f"Box coordinates out of range [0, 1]"

    avg_positives = total_positives / num_samples
    print(f"\nAverage positive anchors per sample: {avg_positives:.2f}")
    print(f"Total anchors per sample: 896")
    print(f"Positive ratio: {avg_positives/896*100:.2f}%")

    print("\n[PASS] Anchor encoding test passed!")


def test_dataloader_integration():
    """Test DataLoader integration."""
    print("\n\nTesting DataLoader integration...")

    from csv_dataloader import get_csv_dataloader

    loader = get_csv_dataloader(
        csv_path="data/splits/train.csv",
        root_dir="data/raw/blazeface",
        batch_size=4,
        shuffle=False,
        num_workers=0,
        target_size=(128, 128),
        augment=False
    )

    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")

    # Get one batch
    batch = next(iter(loader))

    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch anchor_targets shape: {batch['anchor_targets'].shape}")

    # Verify batch dimensions
    assert batch['image'].shape == (4, 3, 128, 128), "Incorrect image batch shape"
    assert batch['anchor_targets'].shape == (4, 896, 5), "Incorrect anchor targets batch shape"

    print("\n[PASS] DataLoader integration test passed!")


def test_augmentation():
    """Test augmentation preserves MediaPipe format."""
    print("\n\nTesting augmentation...")

    dataset_no_aug = CSVDetectorDataset(
        csv_path="data/splits/train.csv",
        root_dir="data/raw/blazeface",
        target_size=(128, 128),
        augment=False
    )

    dataset_with_aug = CSVDetectorDataset(
        csv_path="data/splits/train.csv",
        root_dir="data/raw/blazeface",
        target_size=(128, 128),
        augment=True
    )

    # Test same sample with and without augmentation
    for i in range(5):
        sample_no_aug = dataset_no_aug[i]
        sample_with_aug = dataset_with_aug[i]

        # Both should have positive anchors
        pos_no_aug = (sample_no_aug['anchor_targets'][:, 0] > 0.5).sum()
        pos_with_aug = (sample_with_aug['anchor_targets'][:, 0] > 0.5).sum()

        # Verify format is preserved
        if pos_with_aug > 0:
            boxes = sample_with_aug['anchor_targets'][sample_with_aug['anchor_targets'][:, 0] > 0.5, 1:]
            assert torch.all(boxes >= 0) and torch.all(boxes <= 1), \
                "Augmentation broke box format"

    print("[PASS] Augmentation test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("CSV DataLoader Validation Tests")
    print("=" * 60)

    try:
        test_csv_conversion()
        test_anchor_encoding()
        test_dataloader_integration()
        test_augmentation()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
