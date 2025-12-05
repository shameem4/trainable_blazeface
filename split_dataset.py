"""
Simple utility to create a train/val split from a CSV annotation file.

Images are kept intact across splits by grouping on the image column
(`image_path` by default).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a CSV annotation file into train/val subsets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/blazeface/fixed_images.csv",
        help="Path to the full annotation CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory where the split CSV files will be saved",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image_path",
        help="Column that identifies a unique image (kept intact across splits)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images to place in the validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="train.csv",
        help="Filename to use for the training split inside the output directory",
    )
    parser.add_argument(
        "--val-name",
        type=str,
        default="val.csv",
        help="Filename to use for the validation split inside the output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.image_column not in df.columns:
        raise ValueError(
            f"Column '{args.image_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort by image column and remove exact duplicate rows before splitting
    df = (
        df.sort_values(args.image_column)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    image_ids = df[args.image_column].unique().tolist()
    if not image_ids:
        raise ValueError("No images found in the CSV.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(image_ids)

    val_fraction = min(max(args.val_fraction, 0.0), 1.0)
    n_val = int(round(len(image_ids) * val_fraction))
    if len(image_ids) > 1:
        if n_val == 0:
            n_val = 1
        elif n_val >= len(image_ids):
            n_val = len(image_ids) - 1
    else:
        n_val = len(image_ids)
    val_ids = set(image_ids[:n_val])

    val_df = df[df[args.image_column].isin(val_ids)].reset_index(drop=True)
    train_df = df[~df[args.image_column].isin(val_ids)].reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / args.train_name
    val_path = output_dir / args.val_name

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    total_images = len(image_ids)
    print(
        f"Split {total_images} unique images "
        f"-> train: {len(train_df[args.image_column].unique())} "
        f"({len(train_df)} rows), "
        f"val: {len(val_df[args.image_column].unique())} "
        f"({len(val_df)} rows)"
    )
    print(f"Train CSV saved to: {train_path}")
    print(f"Val CSV saved to:   {val_path}")


if __name__ == "__main__":
    main()
