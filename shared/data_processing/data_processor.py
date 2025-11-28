"""
Data processor for converting raw annotation data into preprocessed NPY metadata files.

This script processes detector and landmarker datasets from data/raw/ and creates
train/validation NPY files in data/preprocessed/.

Features:
- Metadata-only approach: stores image paths + annotations, not images
- 10-100x faster preprocessing (no image loading)
- 10-100x smaller files (~1-10MB instead of ~1-10GB)
- Unified processing for COCO, CSV, and PTS formats
- Progress tracking and error logging
- Uses existing decoders from shared.data_decoder
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import traceback
import argparse

# Import existing decoders
try:
    from shared.data_decoder.coco_decoder import decode_coco_annotation
    from shared.data_decoder.csv_decoder import decode_csv_annotation
    from shared.data_decoder.pts_decoder import decode_pts_annotation
except ImportError:
    import sys
    script_dir = Path(__file__).parent.resolve()
    decoder_dir = script_dir.parent / 'data_decoder'
    sys.path.insert(0, str(decoder_dir))
    from coco_decoder import decode_coco_annotation  # type: ignore
    from csv_decoder import decode_csv_annotation  # type: ignore
    from pts_decoder import decode_pts_annotation  # type: ignore




class DataProcessor:
    """Processes raw data into preprocessed NPY metadata files for training."""

    def __init__(self, raw_data_dir: str = 'data/raw',
                 output_dir: str = 'data/preprocessed'):
        """
        Initialize data processor.

        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory for preprocessed output
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_detector_data(self, train_split: float = 0.8) -> None:
        """
        Process detector datasets into NPY files.
        Automatically detects COCO, CSV, and other formats.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing detector data...")
        detector_dir = self.raw_data_dir / 'detector'

        all_data = []

        # Find all annotation files
        annotation_files = []
        annotation_files.extend(detector_dir.glob('**/_annotations.coco.json'))
        annotation_files.extend(detector_dir.glob('**/*_annotations.csv'))

        for ann_file in annotation_files:
            # Determine type and process
            if ann_file.name.endswith('.json'):
                print(f"  Processing COCO: {ann_file.parent.name}...")
                data = self._process_annotations(ann_file, ann_file.parent, 'coco', 'detector')
            elif ann_file.name.endswith('.csv'):
                print(f"  Processing CSV: {ann_file.name}...")
                data = self._process_annotations(ann_file, ann_file.parent, 'csv', 'detector')
            else:
                continue

            all_data.extend(data)
            print(f"    Found {len(data)} samples")

        # Also check for PTS files (if any exist in detector folder)
        pts_files = list(detector_dir.glob('**/*.pts'))
        if pts_files:
            print(f"  Processing PTS files...")
            data = self._process_annotations(None, detector_dir, 'pts', 'detector')
            all_data.extend(data)
            print(f"    Found {len(data)} samples")

        print(f"Total: {len(all_data)} detector samples")

        # Split and save
        self._split_and_save(all_data, train_split, 'detector')


    def _process_annotations(self, ann_file: Path, image_dir: Path,
                            format_type: str, data_type: str) -> List[Dict]:
        """
        Unified annotation processor for all formats and data types.

        Args:
            ann_file: Annotation file path (or None for PTS which are per-image)
            image_dir: Directory containing images
            format_type: 'coco', 'csv', or 'pts'
            data_type: 'detector', 'landmarker', or 'teacher'

        Returns:
            List of processed data dicts
        """
        data = []

        if format_type == 'coco':
            # COCO format
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)

            for img_info in coco_data['images']:
                image_path = image_dir / img_info['file_name']
                if not image_path.exists():
                    continue

                try:
                    annotations = decode_coco_annotation(str(ann_file), img_info['file_name'])
                    if not annotations:
                        continue

                    sample = self._extract_sample_data(annotations[0], str(image_path), data_type)
                    if sample:
                        data.append(sample)

                except Exception as e:
                    print(f"    [WARNING] {img_info['file_name']}: {e}", file=sys.stderr)

        elif format_type == 'csv':
            # CSV format - group by image
            with open(ann_file, 'r') as f:
                rows = list(csv.DictReader(f))

            image_groups = {}
            for row in rows:
                img_name = row.get('image_path') or row.get('filename')
                if img_name and img_name not in image_groups:
                    image_groups[img_name] = True

            for img_name in image_groups.keys():
                image_path = image_dir / img_name
                if not image_path.exists():
                    continue

                try:
                    annotations = decode_csv_annotation(str(ann_file), img_name)
                    if not annotations:
                        continue

                    sample = self._extract_sample_data(annotations[0], str(image_path), data_type)
                    if sample:
                        data.append(sample)

                except Exception as e:
                    print(f"    [WARNING] {img_name}: {e}", file=sys.stderr)

        elif format_type == 'pts':
            # PTS format - one file per image
            pts_files = list(image_dir.glob('**/*.pts'))

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

                try:
                    annotations = decode_pts_annotation(str(pts_file), image_path.name)
                    if not annotations:
                        continue

                    sample = self._extract_sample_data(annotations[0], str(image_path), data_type)
                    if sample:
                        data.append(sample)

                except Exception as e:
                    print(f"    [WARNING] {pts_file.name}: {e}", file=sys.stderr)

        return data

    def _extract_sample_data(self, annotation: Dict, image_path: str, data_type: str) -> Dict:
        """
        Extract relevant data from annotation based on data type.

        Args:
            annotation: Decoded annotation dict
            image_path: Path to image file
            data_type: 'detector', 'landmarker', or 'teacher'

        Returns:
            Sample dict or None if annotation doesn't match data type
        """
        sample = {'path': image_path}

        if data_type == 'detector':
            # Detector needs bboxes
            if 'bbox' in annotation:
                sample['bboxes'] = [annotation['bbox']]  # Wrap in list
            else:
                return None  # No bbox available

        elif data_type == 'landmarker':
            # Landmarker needs keypoints
            if 'keypoints' in annotation:
                kpts = np.array(annotation['keypoints']).reshape(-1, 3)
                sample['keypoints'] = kpts
            else:
                return None  # No keypoints available

        elif data_type == 'teacher':
            # Teacher needs bbox (or compute from keypoints)
            if 'bbox' in annotation:
                sample['bbox'] = annotation['bbox']
            elif 'keypoints' in annotation:
                # Compute bbox from keypoints with 10% padding
                kpts = np.array(annotation['keypoints']).reshape(-1, 3)
                x_coords, y_coords = kpts[:, 0], kpts[:, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                padding_x = (x_max - x_min) * 0.1
                padding_y = (y_max - y_min) * 0.1
                sample['bbox'] = [
                    max(0, x_min - padding_x),
                    max(0, y_min - padding_y),
                    (x_max - x_min) + 2 * padding_x,
                    (y_max - y_min) + 2 * padding_y
                ]
            else:
                return None  # No bbox or keypoints available

        return sample

    def _split_and_save(self, all_data: List[Dict], train_split: float,
                       data_type: str) -> None:
        """Split data and save to NPY metadata files."""
        n_samples = len(all_data)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Determine data keys
        is_detector = 'bboxes' in all_data[0]

        # Save train NPY (metadata only)
        train_file = self.output_dir / f'train_{data_type}.npy'
        if is_detector:
            train_metadata = {
                'bboxes': np.array([all_data[i]['bboxes'] for i in train_indices], dtype=object),
                'image_paths': np.array([all_data[i]['path'] for i in train_indices])
            }
        else:
            train_metadata = {
                'keypoints': np.array([all_data[i]['keypoints'] for i in train_indices], dtype=object),
                'image_paths': np.array([all_data[i]['path'] for i in train_indices])
            }
        np.save(train_file, train_metadata, allow_pickle=True)
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPY (metadata only)
        val_file = self.output_dir / f'val_{data_type}.npy'
        if is_detector:
            val_metadata = {
                'bboxes': np.array([all_data[i]['bboxes'] for i in val_indices], dtype=object),
                'image_paths': np.array([all_data[i]['path'] for i in val_indices])
            }
        else:
            val_metadata = {
                'keypoints': np.array([all_data[i]['keypoints'] for i in val_indices], dtype=object),
                'image_paths': np.array([all_data[i]['path'] for i in val_indices])
            }
        np.save(val_file, val_metadata, allow_pickle=True)
        print(f"Saved {len(val_indices)} validation samples to {val_file}")

    def process_landmarker_data(self, train_split: float = 0.8) -> None:
        """
        Process landmarker datasets into NPY files.
        Automatically detects PTS, COCO, and other formats.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing landmarker data...")
        landmarker_dir = self.raw_data_dir / 'landmarker'

        all_data = []

        # Find all annotation files
        annotation_files = []
        annotation_files.extend(landmarker_dir.glob('**/_annotations.coco.json'))
        annotation_files.extend(landmarker_dir.glob('**/*_annotations.csv'))

        for ann_file in annotation_files:
            if ann_file.name.endswith('.json'):
                print(f"  Processing COCO: {ann_file.parent.name}...")
                data = self._process_annotations(ann_file, ann_file.parent, 'coco', 'landmarker')
            elif ann_file.name.endswith('.csv'):
                print(f"  Processing CSV: {ann_file.name}...")
                data = self._process_annotations(ann_file, ann_file.parent, 'csv', 'landmarker')
            else:
                continue

            all_data.extend(data)
            print(f"    Found {len(data)} samples")

        # Check for PTS files
        pts_files = list(landmarker_dir.glob('**/*.pts'))
        if pts_files:
            print(f"  Processing PTS files...")
            data = self._process_annotations(None, landmarker_dir, 'pts', 'landmarker')
            all_data.extend(data)
            print(f"    Found {len(data)} samples")

        print(f"Total: {len(all_data)} landmarker samples")

        # Split and save
        self._split_and_save(all_data, train_split, 'landmarker')

    def process_teacher_data(self, train_split: float = 0.8) -> None:
        """
        Process teacher datasets (compute bboxes for autoencoder).
        Collects data from both detector and landmarker folders.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing teacher data (bboxes for autoencoder)...")

        all_data = []

        # Collect from detector folder
        detector_dir = self.raw_data_dir / 'detector'
        if detector_dir.exists():
            # Find all annotation files
            annotation_files = []
            annotation_files.extend(detector_dir.glob('**/_annotations.coco.json'))
            annotation_files.extend(detector_dir.glob('**/*_annotations.csv'))

            for ann_file in annotation_files:
                if ann_file.name.endswith('.json'):
                    print(f"  Processing detector COCO: {ann_file.parent.name}...")
                    data = self._process_annotations(ann_file, ann_file.parent, 'coco', 'teacher')
                elif ann_file.name.endswith('.csv'):
                    print(f"  Processing detector CSV: {ann_file.name}...")
                    data = self._process_annotations(ann_file, ann_file.parent, 'csv', 'teacher')
                else:
                    continue

                all_data.extend(data)
                print(f"    Found {len(data)} samples")

        # Collect from landmarker folder
        landmarker_dir = self.raw_data_dir / 'landmarker'
        if landmarker_dir.exists():
            # PTS datasets
            pts_files = list(landmarker_dir.glob('**/*.pts'))
            if pts_files:
                print(f"  Processing landmarker PTS files...")
                data = self._process_annotations(None, landmarker_dir, 'pts', 'teacher')
                all_data.extend(data)
                print(f"    Found {len(data)} samples")

            # COCO landmarker datasets
            annotation_files = []
            annotation_files.extend(landmarker_dir.glob('**/_annotations.coco.json'))
            annotation_files.extend(landmarker_dir.glob('**/*_annotations.csv'))

            for ann_file in annotation_files:
                if ann_file.name.endswith('.json'):
                    print(f"  Processing landmarker COCO: {ann_file.parent.name}...")
                    data = self._process_annotations(ann_file, ann_file.parent, 'coco', 'teacher')
                elif ann_file.name.endswith('.csv'):
                    print(f"  Processing landmarker CSV: {ann_file.name}...")
                    data = self._process_annotations(ann_file, ann_file.parent, 'csv', 'teacher')
                else:
                    continue

                all_data.extend(data)
                print(f"    Found {len(data)} samples")

        if not all_data:
            print("No teacher data found!")
            return

        print(f"Total: {len(all_data)} teacher samples")

        # Split and save - teacher data uses 'bbox' (singular) not 'bboxes'
        n_samples = len(all_data)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Save train NPY (metadata only)
        train_file = self.output_dir / 'train_teacher.npy'
        train_metadata = {
            'image_paths': np.array([all_data[i]['path'] for i in train_indices]),
            'bboxes': np.array([all_data[i]['bbox'] for i in train_indices], dtype=object)
        }
        np.save(train_file, train_metadata, allow_pickle=True)
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPY (metadata only)
        val_file = self.output_dir / 'val_teacher.npy'
        val_metadata = {
            'image_paths': np.array([all_data[i]['path'] for i in val_indices]),
            'bboxes': np.array([all_data[i]['bbox'] for i in val_indices], dtype=object)
        }
        np.save(val_file, val_metadata, allow_pickle=True)
        print(f"Saved {len(val_indices)} validation samples to {val_file}")

    def process_all(self, train_split: float = 0.8,
                   include_teacher: bool = True) -> None:
        """
        Process all datasets (detector, landmarker, and teacher).

        Args:
            train_split: Fraction of data to use for training
            include_teacher: Whether to process teacher data for autoencoder
        """
        print("=" * 60)
        print("Data Processing Pipeline (Metadata-Only)")
        print("=" * 60)
        print()

        errors = []

        # Process detector data
        try:
            self.process_detector_data(train_split)
        except Exception as e:
            error_msg = f"Detector processing failed: {type(e).__name__}: {e}"
            print(f"\n[ERROR] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            errors.append(error_msg)

        print()

        # Process landmarker data
        try:
            self.process_landmarker_data(train_split)
        except Exception as e:
            error_msg = f"Landmarker processing failed: {type(e).__name__}: {e}"
            print(f"\n[ERROR] {error_msg}", file=sys.stderr)
            traceback.print_exc()
            errors.append(error_msg)

        # Process teacher data
        if include_teacher:
            print()
            try:
                self.process_teacher_data(train_split)
            except Exception as e:
                error_msg = f"Teacher processing failed: {type(e).__name__}: {e}"
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                errors.append(error_msg)

        print()
        print("=" * 60)
        if errors:
            print("Processing completed WITH ERRORS!")
            print("=" * 60)
            print("\nError Summary:", file=sys.stderr)
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}", file=sys.stderr)
            print("\nCheck error messages above for details.", file=sys.stderr)
        else:
            print("Processing complete!")
            print("=" * 60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print("=" * 60)


def main():
    """Main entry point for data processing."""
    parser = argparse.ArgumentParser(
        description='Process ear detection and landmark datasets into NPZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  python data_processor.py --all

  # Process only detector data
  python data_processor.py --detector

  # Process detector and landmarker (no teacher)
  python data_processor.py --detector --landmarker

  # Process with custom settings
  python data_processor.py --all --split 0.85
        """
    )

    # Data selection arguments
    data_group = parser.add_argument_group('Data Selection')
    data_group.add_argument('--all', action='store_true',
                           help='Process all datasets (detector, landmarker, teacher)')
    data_group.add_argument('--detector', action='store_true',
                           help='Process detector data (ear bounding boxes)')
    data_group.add_argument('--landmarker', action='store_true',
                           help='Process landmarker data (ear keypoints)')
    data_group.add_argument('--teacher', action='store_true',
                           help='Process teacher data (bboxes for autoencoder)')

    # Configuration arguments
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--split', type=float, default=0.8,
                             help='Train/validation split ratio (default: 0.8)')
    config_group.add_argument('--input-dir', type=str, default='data/raw',
                             help='Input directory containing raw data (default: data/raw)')
    config_group.add_argument('--output-dir', type=str, default='data/preprocessed',
                             help='Output directory for NPY files (default: data/preprocessed)')

    args = parser.parse_args()

    # Validate arguments
    if not (args.all or args.detector or args.landmarker or args.teacher):
        parser.error('Must specify at least one of: --all, --detector, --landmarker, --teacher')

    if args.split <= 0 or args.split >= 1:
        parser.error('--split must be between 0 and 1')

    # Create processor
    processor = DataProcessor(
        raw_data_dir=args.input_dir,
        output_dir=args.output_dir
    )

    # Determine what to process
    if args.all:
        # Process everything
        processor.process_all(train_split=args.split, include_teacher=True)
    else:
        # Process selected datasets
        print("=" * 60)
        print("Data Processing Pipeline (Metadata-Only)")
        print("=" * 60)
        print()

        errors = []

        if args.detector:
            try:
                processor.process_detector_data(args.split)
            except Exception as e:
                error_msg = f"Detector processing failed: {type(e).__name__}: {e}"
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                errors.append(error_msg)
            print()

        if args.landmarker:
            try:
                processor.process_landmarker_data(args.split)
            except Exception as e:
                error_msg = f"Landmarker processing failed: {type(e).__name__}: {e}"
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                errors.append(error_msg)
            print()

        if args.teacher:
            try:
                processor.process_teacher_data(args.split)
            except Exception as e:
                error_msg = f"Teacher processing failed: {type(e).__name__}: {e}"
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                errors.append(error_msg)
            print()

        # Print summary
        print("=" * 60)
        if errors:
            print("Processing completed WITH ERRORS!")
            print("=" * 60)
            print("\nError Summary:", file=sys.stderr)
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}", file=sys.stderr)
            print("\nCheck error messages above for details.", file=sys.stderr)
        else:
            print("Processing complete!")
            print("=" * 60)
        print(f"Output directory: {processor.output_dir.absolute()}")
        print("=" * 60)


if __name__ == '__main__':
    main()
