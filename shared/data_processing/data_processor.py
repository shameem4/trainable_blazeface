"""
Data processor for converting raw annotation data into preprocessed NPZ files.

This script processes detector and landmarker datasets from data/raw/ and creates
train/validation NPZ files in data/preprocessed/.

Features:
- Parallel processing using multiprocessing
- Memory-efficient batch processing
- Progress tracking
- Uses existing decoders from shared.data_decoder
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
import gc

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


# Worker functions for multiprocessing (must be at module level)
def process_coco_image(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Process a single COCO image in parallel."""
    annotation_file, image_path, filename = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_coco_annotation(annotation_file, filename)
        if not annotations:
            return None

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        bboxes = [ann['bbox'] for ann in annotations if 'bbox' in ann]

        return {
            'image': img_array,
            'bboxes': bboxes,
            'path': str(image_path)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_coco_landmarker_image(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Process a single COCO landmarker image in parallel."""
    annotation_file, image_path, filename = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_coco_annotation(annotation_file, filename)
        if not annotations or 'keypoints' not in annotations[0]:
            return None

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        kpts = annotations[0]['keypoints']
        keypoints = np.array(kpts).reshape(-1, 3)

        return {
            'image': img_array,
            'keypoints': keypoints,
            'path': str(image_path)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_pts_file(args: Tuple[str, str]) -> Optional[Dict]:
    """Process a single PTS file in parallel."""
    pts_file, image_path = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_pts_annotation(pts_file, Path(image_path).name)
        if not annotations or 'keypoints' not in annotations[0]:
            return None

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        kpts = annotations[0]['keypoints']
        keypoints = np.array(kpts).reshape(-1, 3)

        return {
            'image': img_array,
            'keypoints': keypoints,
            'path': str(image_path)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_csv_image(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Process a single CSV image in parallel."""
    csv_file, image_path, img_name = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_csv_annotation(csv_file, img_name)
        if not annotations:
            return None

        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        bboxes = [ann['bbox'] for ann in annotations if 'bbox' in ann]

        return {
            'image': img_array,
            'bboxes': bboxes,
            'path': str(image_path)
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


class DataProcessor:
    """Processes raw data into preprocessed NPZ files for training."""

    def __init__(self, raw_data_dir: str = 'data/raw',
                 output_dir: str = 'data/preprocessed',
                 batch_size: int = 100,
                 num_workers: Optional[int] = None):
        """
        Initialize data processor.

        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory for preprocessed output
            batch_size: Batch size for memory-efficient processing
            num_workers: Number of parallel workers (None = CPU count)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()

    def process_detector_data(self, train_split: float = 0.8) -> None:
        """
        Process detector datasets (COCO and CSV formats) into NPZ files.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing detector data...")
        print(f"Using {self.num_workers} workers for parallel processing")
        detector_dir = self.raw_data_dir / 'detector'

        # Collect all tasks
        tasks = []

        # COCO datasets
        coco_dirs = [d for d in detector_dir.glob('*')
                     if d.is_dir() and 'coco' in d.name.lower()]
        for coco_dir in coco_dirs:
            for split in ['train', 'test', 'valid']:
                annotation_file = coco_dir / split / '_annotations.coco.json'
                if annotation_file.exists():
                    tasks.extend(self._collect_coco_tasks(annotation_file, coco_dir / split))

        # CSV datasets
        csv_files = list(detector_dir.glob('**/*_annotations.csv'))
        for csv_file in csv_files:
            tasks.extend(self._collect_csv_tasks(csv_file, csv_file.parent))

        print(f"Found {len(tasks)} images to process")

        # Process in parallel with batches
        all_data = self._process_tasks_in_batches(tasks, process_coco_image)

        # Split and save
        self._split_and_save(all_data, train_split, 'detector')

    def _collect_coco_tasks(self, annotation_file: Path,
                           image_dir: Path) -> List[Tuple]:
        """Collect COCO processing tasks."""
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        tasks = []
        for img_info in coco_data['images']:
            image_path = image_dir / img_info['file_name']
            tasks.append((str(annotation_file), str(image_path), img_info['file_name']))
        return tasks

    def _collect_csv_tasks(self, csv_file: Path, image_dir: Path) -> List[Tuple]:
        """Collect CSV processing tasks."""
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group by image
        image_groups = {}
        for row in rows:
            img_name = row.get('image_path') or row.get('filename')
            if img_name and img_name not in image_groups:
                image_groups[img_name] = True

        tasks = []
        for img_name in image_groups.keys():
            image_path = image_dir / img_name
            tasks.append((str(csv_file), str(image_path), img_name))
        return tasks

    def _process_tasks_in_batches(self, tasks: List, worker_func) -> List[Dict]:
        """Process tasks in batches for memory efficiency."""
        all_data = []
        total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(tasks))
            batch_tasks = tasks[start_idx:end_idx]

            print(f"  Processing batch {batch_idx + 1}/{total_batches} "
                  f"({len(batch_tasks)} images)...")

            # Process batch in parallel
            with Pool(self.num_workers) as pool:
                results = pool.map(worker_func, batch_tasks)

            # Filter out None results and collect
            batch_data = [r for r in results if r is not None]
            all_data.extend(batch_data)

            # Force garbage collection after each batch
            gc.collect()

        print(f"Successfully processed {len(all_data)} images")
        return all_data

    def _split_and_save(self, all_data: List[Dict], train_split: float,
                       data_type: str) -> None:
        """Split data and save to NPZ files."""
        n_samples = len(all_data)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Determine data keys
        is_detector = 'bboxes' in all_data[0]

        # Save train NPZ
        train_file = self.output_dir / f'train_{data_type}.npz'
        if is_detector:
            np.savez_compressed(
                train_file,
                images=np.array([all_data[i]['image'] for i in train_indices]),
                bboxes=np.array([all_data[i]['bboxes'] for i in train_indices],
                               dtype=object),
                image_paths=np.array([all_data[i]['path'] for i in train_indices])
            )
        else:
            np.savez_compressed(
                train_file,
                images=np.array([all_data[i]['image'] for i in train_indices]),
                keypoints=np.array([all_data[i]['keypoints'] for i in train_indices]),
                image_paths=np.array([all_data[i]['path'] for i in train_indices])
            )
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPZ
        val_file = self.output_dir / f'val_{data_type}.npz'
        if is_detector:
            np.savez_compressed(
                val_file,
                images=np.array([all_data[i]['image'] for i in val_indices]),
                bboxes=np.array([all_data[i]['bboxes'] for i in val_indices],
                               dtype=object),
                image_paths=np.array([all_data[i]['path'] for i in val_indices])
            )
        else:
            np.savez_compressed(
                val_file,
                images=np.array([all_data[i]['image'] for i in val_indices]),
                keypoints=np.array([all_data[i]['keypoints'] for i in val_indices]),
                image_paths=np.array([all_data[i]['path'] for i in val_indices])
            )
        print(f"Saved {len(val_indices)} validation samples to {val_file}")

    def process_landmarker_data(self, train_split: float = 0.8) -> None:
        """
        Process landmarker datasets (PTS and COCO formats) into NPZ files.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing landmarker data...")
        print(f"Using {self.num_workers} workers for parallel processing")
        landmarker_dir = self.raw_data_dir / 'landmarker'

        # Collect all tasks
        tasks = []

        # PTS datasets
        pts_collections = ['collectionA', 'collectionB']
        for collection in pts_collections:
            collection_dir = landmarker_dir / collection
            if collection_dir.exists():
                print(f"  Collecting tasks from {collection}...")
                tasks.extend(self._collect_pts_tasks(collection_dir))

        # COCO landmarker datasets
        coco_dirs = [d for d in landmarker_dir.glob('*')
                     if d.is_dir() and 'coco' in d.name.lower()]
        for coco_dir in coco_dirs:
            for split in ['train', 'test', 'valid']:
                annotation_file = coco_dir / split / '_annotations.coco.json'
                if annotation_file.exists():
                    tasks.extend(self._collect_coco_tasks(annotation_file,
                                                         coco_dir / split))

        print(f"Found {len(tasks)} images to process")

        # Determine task type and worker function
        if tasks and len(tasks[0]) == 2:  # PTS tasks
            worker_func = process_pts_file
        else:  # COCO tasks
            worker_func = process_coco_landmarker_image

        # Process in parallel with batches
        all_data = self._process_tasks_in_batches(tasks, worker_func)

        # Split and save
        self._split_and_save(all_data, train_split, 'landmarker')

    def _collect_pts_tasks(self, collection_dir: Path) -> List[Tuple]:
        """Collect PTS processing tasks."""
        pts_files = list(collection_dir.glob('*.pts'))
        tasks = []

        for pts_file in pts_files:
            # Find corresponding image
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_path = pts_file.with_suffix(ext)
                if potential_path.exists():
                    image_path = potential_path
                    break

            if image_path:
                tasks.append((str(pts_file), str(image_path)))

        return tasks

    def process_all(self, train_split: float = 0.8) -> None:
        """
        Process all datasets (detector and landmarker).

        Args:
            train_split: Fraction of data to use for training
        """
        print("=" * 60)
        print("Data Processing Pipeline (Parallel & Memory-Efficient)")
        print("=" * 60)
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.num_workers}")
        print()

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
    # Adjust batch_size and num_workers based on available memory
    processor = DataProcessor(
        batch_size=100,  # Process 100 images at a time
        num_workers=None  # Use all available CPU cores
    )
    processor.process_all(train_split=0.8)


if __name__ == '__main__':
    main()
