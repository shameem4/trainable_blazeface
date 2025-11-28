"""
Data processor for converting raw annotation data into preprocessed NPZ files.

This script processes detector and landmarker datasets from data/raw/ and creates
train/validation NPZ files in data/preprocessed/.

Uses existing decoders from shared.data_decoder for reading annotations.
"""

import os
import json
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional

# Import existing decoders
try:
    from shared.data_decoder.decoder import decode_annotation
    from shared.data_decoder.coco_decoder import decode_coco_annotation
    from shared.data_decoder.csv_decoder import decode_csv_annotation
    from shared.data_decoder.pts_decoder import decode_pts_annotation
except ImportError:
    import sys
    script_dir = Path(__file__).parent.resolve()
    decoder_dir = script_dir.parent / 'data_decoder'
    sys.path.insert(0, str(decoder_dir))
    from decoder import decode_annotation  # type: ignore
    from coco_decoder import decode_coco_annotation  # type: ignore
    from csv_decoder import decode_csv_annotation  # type: ignore
    from pts_decoder import decode_pts_annotation  # type: ignore


class DataProcessor:
    """Processes raw data into preprocessed NPZ files for training."""

    def __init__(self, raw_data_dir: str = 'data/raw', output_dir: str = 'data/preprocessed'):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_detector_data(self, train_split: float = 0.8) -> None:
        """
        Process detector datasets (COCO and CSV formats) into NPZ files.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing detector data...")
        detector_dir = self.raw_data_dir / 'detector'

        all_images = []
        all_bboxes = []
        all_image_paths = []

        # Process COCO datasets
        coco_dirs = [d for d in detector_dir.glob('*') if d.is_dir() and 'coco' in d.name.lower()]
        for coco_dir in coco_dirs:
            print(f"  Processing {coco_dir.name}...")
            for split in ['train', 'test', 'valid']:
                annotation_file = coco_dir / split / '_annotations.coco.json'
                if annotation_file.exists():
                    self._process_coco_dataset(annotation_file, coco_dir / split,
                                              all_images, all_bboxes, all_image_paths)

        # Process CSV datasets
        csv_files = list(detector_dir.glob('**/*_annotations.csv'))
        for csv_file in csv_files:
            print(f"  Processing {csv_file.name}...")
            self._process_csv_dataset(csv_file, csv_file.parent,
                                     all_images, all_bboxes, all_image_paths)

        # Split into train and validation
        n_samples = len(all_images)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Save train NPZ
        train_file = self.output_dir / 'train_detector.npz'
        np.savez_compressed(
            train_file,
            images=np.array([all_images[i] for i in train_indices]),
            bboxes=np.array([all_bboxes[i] for i in train_indices], dtype=object),
            image_paths=np.array([all_image_paths[i] for i in train_indices])
        )
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPZ
        val_file = self.output_dir / 'val_detector.npz'
        np.savez_compressed(
            val_file,
            images=np.array([all_images[i] for i in val_indices]),
            bboxes=np.array([all_bboxes[i] for i in val_indices], dtype=object),
            image_paths=np.array([all_image_paths[i] for i in val_indices])
        )
        print(f"Saved {len(val_indices)} validation samples to {val_file}")

    def _process_coco_dataset(self, annotation_file: Path, image_dir: Path,
                             all_images: List, all_bboxes: List, all_image_paths: List) -> None:
        """Process a COCO format dataset."""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        for img_info in coco_data['images']:
            image_path = image_dir / img_info['file_name']
            if not image_path.exists():
                continue

            # Use existing decoder
            annotations = decode_coco_annotation(str(annotation_file), img_info['file_name'])

            if annotations:
                # Load image
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)

                # Extract bboxes
                bboxes = [ann['bbox'] for ann in annotations if 'bbox' in ann]

                all_images.append(img_array)
                all_bboxes.append(bboxes)
                all_image_paths.append(str(image_path))

    def _process_csv_dataset(self, csv_file: Path, image_dir: Path,
                            all_images: List, all_bboxes: List, all_image_paths: List) -> None:
        """Process a CSV format dataset."""
        # Read CSV to get unique images
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group by image
        image_groups = {}
        for row in rows:
            img_name = row.get('image_path') or row.get('filename')
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(row)

        for img_name, _ in image_groups.items():
            image_path = image_dir / img_name
            if not image_path.exists():
                continue

            # Use existing decoder
            annotations = decode_csv_annotation(str(csv_file), img_name)

            if annotations:
                # Load image
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)

                # Extract bboxes
                bboxes = [ann['bbox'] for ann in annotations if 'bbox' in ann]

                all_images.append(img_array)
                all_bboxes.append(bboxes)
                all_image_paths.append(str(image_path))

    def process_landmarker_data(self, train_split: float = 0.8) -> None:
        """
        Process landmarker datasets (PTS and COCO formats) into NPZ files.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing landmarker data...")
        landmarker_dir = self.raw_data_dir / 'landmarker'

        all_images = []
        all_keypoints = []
        all_image_paths = []

        # Process PTS datasets (collectionA, collectionB)
        pts_collections = ['collectionA', 'collectionB']
        for collection in pts_collections:
            collection_dir = landmarker_dir / collection
            if collection_dir.exists():
                print(f"  Processing {collection}...")
                self._process_pts_collection(collection_dir, all_images, all_keypoints, all_image_paths)

        # Process COCO landmarker datasets
        coco_dirs = [d for d in landmarker_dir.glob('*') if d.is_dir() and 'coco' in d.name.lower()]
        for coco_dir in coco_dirs:
            print(f"  Processing {coco_dir.name}...")
            for split in ['train', 'test', 'valid']:
                annotation_file = coco_dir / split / '_annotations.coco.json'
                if annotation_file.exists():
                    self._process_coco_landmarker(annotation_file, coco_dir / split,
                                                 all_images, all_keypoints, all_image_paths)

        # Split into train and validation
        n_samples = len(all_images)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Save train NPZ
        train_file = self.output_dir / 'train_landmarker.npz'
        np.savez_compressed(
            train_file,
            images=np.array([all_images[i] for i in train_indices]),
            keypoints=np.array([all_keypoints[i] for i in train_indices]),
            image_paths=np.array([all_image_paths[i] for i in train_indices])
        )
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPZ
        val_file = self.output_dir / 'val_landmarker.npz'
        np.savez_compressed(
            val_file,
            images=np.array([all_images[i] for i in val_indices]),
            keypoints=np.array([all_keypoints[i] for i in val_indices]),
            image_paths=np.array([all_image_paths[i] for i in val_indices])
        )
        print(f"Saved {len(val_indices)} validation samples to {val_file}")

    def _process_pts_collection(self, collection_dir: Path,
                               all_images: List, all_keypoints: List, all_image_paths: List) -> None:
        """Process a PTS collection directory."""
        pts_files = list(collection_dir.glob('*.pts'))

        for pts_file in pts_files:
            # Find corresponding image
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = pts_file.with_suffix(ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

            if not image_path:
                continue

            # Use existing decoder
            annotations = decode_pts_annotation(str(pts_file), image_path.name)

            if annotations and 'keypoints' in annotations[0]:
                # Load image
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)

                # Extract keypoints (reshape to Nx3 format: x, y, visibility)
                kpts = annotations[0]['keypoints']
                # keypoints are [x1, y1, v1, x2, y2, v2, ...]
                keypoints = np.array(kpts).reshape(-1, 3)

                all_images.append(img_array)
                all_keypoints.append(keypoints)
                all_image_paths.append(str(image_path))

    def _process_coco_landmarker(self, annotation_file: Path, image_dir: Path,
                                all_images: List, all_keypoints: List, all_image_paths: List) -> None:
        """Process COCO format landmarker dataset with keypoints."""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        for img_info in coco_data['images']:
            image_path = image_dir / img_info['file_name']
            if not image_path.exists():
                continue

            # Use existing decoder
            annotations = decode_coco_annotation(str(annotation_file), img_info['file_name'])

            if annotations and 'keypoints' in annotations[0]:
                # Load image
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img)

                # Extract keypoints
                kpts = annotations[0]['keypoints']
                keypoints = np.array(kpts).reshape(-1, 3)

                all_images.append(img_array)
                all_keypoints.append(keypoints)
                all_image_paths.append(str(image_path))

    def process_all(self, train_split: float = 0.8) -> None:
        """
        Process all datasets (detector and landmarker).

        Args:
            train_split: Fraction of data to use for training
        """
        print("=" * 60)
        print("Data Processing Pipeline")
        print("=" * 60)

        self.process_detector_data(train_split)
        print()
        self.process_landmarker_data(train_split)

        print()
        print("=" * 60)
        print("Processing complete!")
        print(f"Output directory: {self.output_dir.absolute()}")
        print("=" * 60)


def main():
    """Main entry point for data processing."""
    processor = DataProcessor()
    processor.process_all(train_split=0.8)


if __name__ == '__main__':
    main()
