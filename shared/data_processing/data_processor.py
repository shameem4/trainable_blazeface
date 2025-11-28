"""
Data processor for converting raw annotation data into preprocessed NPZ files.

This script processes detector and landmarker datasets from data/raw/ and creates
train/validation NPZ files in data/preprocessed/.

Features:
- Parallel processing using multiprocessing
- Memory-efficient batch processing
- Progress tracking
- Error logging and propagation
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
from threading import Thread, Lock
from queue import Queue
import gc
import sys
import traceback
import argparse

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
        print(f"\n[ERROR] Failed to process {image_path}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
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
        print(f"\n[ERROR] Failed to process landmarker {image_path}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
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
        print(f"\n[ERROR] Failed to process PTS file {pts_file}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
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
        print(f"\n[ERROR] Failed to process CSV image {image_path}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def process_teacher_image(args: Tuple[str, str, str]) -> Optional[Dict]:
    """
    Process image for teacher autoencoder training.
    Crops ear using bounding box from annotations.
    """
    annotation_file, image_path, filename = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_coco_annotation(annotation_file, filename)
        if not annotations:
            return None

        # Get first bbox (assuming one ear per image for teacher)
        bbox = None
        if 'bbox' in annotations[0]:
            bbox = annotations[0]['bbox']
        elif 'keypoints' in annotations[0]:
            # Compute bbox from keypoints
            kpts = np.array(annotations[0]['keypoints']).reshape(-1, 3)
            x_coords = kpts[:, 0]
            y_coords = kpts[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            # Add padding (10%)
            padding_x = (x_max - x_min) * 0.1
            padding_y = (y_max - y_min) * 0.1
            bbox = [
                max(0, x_min - padding_x),
                max(0, y_min - padding_y),
                (x_max - x_min) + 2 * padding_x,
                (y_max - y_min) + 2 * padding_y
            ]

        if not bbox:
            return None

        # Load and crop image
        img = Image.open(image_path).convert('RGB')
        x, y, w, h = bbox
        # Ensure bbox is within image bounds
        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)

        cropped_img = img.crop((x, y, x2, y2))
        cropped_array = np.array(cropped_img)

        return {
            'image': cropped_array,
            'path': str(image_path),
            'bbox': bbox
        }
    except Exception as e:
        print(f"\n[ERROR] Failed to process teacher image {image_path}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def process_teacher_pts(args: Tuple[str, str]) -> Optional[Dict]:
    """Process PTS file for teacher data (compute bbox from keypoints)."""
    pts_file, image_path = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_pts_annotation(pts_file, Path(image_path).name)
        if not annotations or 'keypoints' not in annotations[0]:
            return None

        # Compute bbox from keypoints
        kpts = np.array(annotations[0]['keypoints']).reshape(-1, 3)
        x_coords = kpts[:, 0]
        y_coords = kpts[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Add padding (10%)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        bbox = [
            max(0, x_min - padding_x),
            max(0, y_min - padding_y),
            (x_max - x_min) + 2 * padding_x,
            (y_max - y_min) + 2 * padding_y
        ]

        # Load and crop image
        img = Image.open(image_path).convert('RGB')
        x, y, w, h = bbox
        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)

        cropped_img = img.crop((x, y, x2, y2))
        cropped_array = np.array(cropped_img)

        return {
            'image': cropped_array,
            'path': str(image_path),
            'bbox': bbox
        }
    except Exception as e:
        print(f"\n[ERROR] Failed to process teacher PTS {pts_file}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def process_teacher_csv(args: Tuple[str, str, str]) -> Optional[Dict]:
    """Process CSV file for teacher data."""
    csv_file, image_path, img_name = args

    if not Path(image_path).exists():
        return None

    try:
        annotations = decode_csv_annotation(csv_file, img_name)
        if not annotations or 'bbox' not in annotations[0]:
            return None

        bbox = annotations[0]['bbox']

        # Load and crop image
        img = Image.open(image_path).convert('RGB')
        x, y, w, h = bbox
        x = max(0, int(x))
        y = max(0, int(y))
        w = int(w)
        h = int(h)
        x2 = min(img.width, x + w)
        y2 = min(img.height, y + h)

        cropped_img = img.crop((x, y, x2, y2))
        cropped_array = np.array(cropped_img)

        return {
            'image': cropped_array,
            'path': str(image_path),
            'bbox': bbox
        }
    except Exception as e:
        print(f"\n[ERROR] Failed to process teacher CSV {image_path}", file=sys.stderr)
        print(f"  Exception: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


class BackgroundDiskWriter:
    """Background thread for writing data to disk while processing continues."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.queue = Queue()
        self.thread = Thread(target=self._write_worker, daemon=True)
        self.running = True
        self.lock = Lock()
        self.temp_files = []
        self.thread.start()

    def _write_worker(self):
        """Background worker that writes data to disk."""
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:  # Poison pill
                    break

                temp_file, data = item
                self._save_temp_file(temp_file, data)

                with self.lock:
                    self.temp_files.append(str(temp_file))

                self.queue.task_done()
            except:
                continue

    def _save_temp_file(self, temp_file: Path, data: List[Dict]) -> None:
        """Save accumulated data to temporary NPZ file."""
        if not data:
            return

        # Determine data structure
        is_detector = 'bboxes' in data[0]
        is_teacher = 'bbox' in data[0] and 'bboxes' not in data[0]

        if is_teacher:
            # Teacher data (single bbox per image)
            np.savez_compressed(
                temp_file,
                images=np.array([d['image'] for d in data], dtype=object),
                image_paths=np.array([d['path'] for d in data]),
                bboxes=np.array([d['bbox'] for d in data], dtype=object)
            )
        elif is_detector:
            # Detector data (multiple bboxes per image)
            np.savez_compressed(
                temp_file,
                images=np.array([d['image'] for d in data], dtype=object),
                bboxes=np.array([d['bboxes'] for d in data], dtype=object),
                image_paths=np.array([d['path'] for d in data])
            )
        else:
            # Landmarker data (keypoints)
            np.savez_compressed(
                temp_file,
                images=np.array([d['image'] for d in data], dtype=object),
                keypoints=np.array([d['keypoints'] for d in data], dtype=object),
                image_paths=np.array([d['path'] for d in data])
            )

    def enqueue_write(self, temp_file: Path, data: List[Dict]):
        """Add data to the write queue (non-blocking)."""
        self.queue.put((temp_file, data))

    def wait_completion(self):
        """Wait for all writes to complete."""
        self.queue.join()

    def shutdown(self):
        """Shutdown the background writer."""
        self.running = False
        self.queue.put(None)  # Poison pill
        self.thread.join()

    def get_temp_files(self) -> List[str]:
        """Get list of written temp files."""
        with self.lock:
            return self.temp_files.copy()


class DataProcessor:
    """Processes raw data into preprocessed NPZ files for training."""

    def __init__(self, raw_data_dir: str = 'data/raw',
                 output_dir: str = 'data/preprocessed',
                 batch_size: int = 100,
                 num_workers: Optional[int] = None,
                 flush_every: int = 5):
        """
        Initialize data processor.

        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory for preprocessed output
            batch_size: Batch size for memory-efficient processing
            num_workers: Number of parallel workers (None = CPU count)
            flush_every: Number of batches before flushing to temp file (default: 5)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_workers = num_workers or cpu_count()
        self.flush_every = flush_every
        self.temp_dir = self.output_dir / '.temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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

        # Separate COCO and CSV tasks
        coco_tasks = [t for t in tasks if '.csv' not in t[0]]
        csv_tasks = [t for t in tasks if '.csv' in t[0]]

        temp_files = []

        # Process COCO tasks
        if coco_tasks:
            print(f"  Processing {len(coco_tasks)} COCO images...")
            coco_files = self._process_tasks_in_batches(coco_tasks, process_coco_image,
                                                        'detector_coco')
            temp_files.extend(coco_files)

        # Process CSV tasks
        if csv_tasks:
            print(f"  Processing {len(csv_tasks)} CSV images...")
            csv_files = self._process_tasks_in_batches(csv_tasks, process_csv_image,
                                                       'detector_csv')
            temp_files.extend(csv_files)

        # Load temp files, shuffle, and save final files
        all_data = self._load_and_merge_temp_files(temp_files)
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

    def _process_tasks_in_batches(self, tasks: List, worker_func,
                                  data_type: str = 'temp') -> List[str]:
        """
        Process tasks in batches with background disk flushing.

        Returns:
            List of temporary file paths containing processed data
        """
        # Start background writer
        bg_writer = BackgroundDiskWriter(self.temp_dir)
        flush_counter = 0

        accumulated_data = []
        total_batches = (len(tasks) + self.batch_size - 1) // self.batch_size
        total_failed = 0
        total_processed = 0

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
            batch_failed = len(results) - len(batch_data)
            total_failed += batch_failed
            total_processed += len(batch_data)

            if batch_failed > 0:
                print(f"    [WARNING] {batch_failed} images failed in this batch",
                      file=sys.stderr)

            accumulated_data.extend(batch_data)

            # Flush to disk every N batches (background)
            if (batch_idx + 1) % self.flush_every == 0 or (batch_idx + 1) == total_batches:
                if accumulated_data:
                    temp_file = self.temp_dir / f'{data_type}_{flush_counter}.npz'
                    print(f"    Enqueuing {len(accumulated_data)} samples for background write to {temp_file.name}")

                    # Enqueue write (non-blocking - processing can continue!)
                    bg_writer.enqueue_write(temp_file, accumulated_data)
                    flush_counter += 1

                    # Clear memory immediately
                    accumulated_data = []
                    gc.collect()

        # Wait for all background writes to complete
        print(f"  Waiting for background writes to complete...")
        bg_writer.wait_completion()
        temp_files = bg_writer.get_temp_files()
        bg_writer.shutdown()

        print(f"Successfully processed {total_processed}/{len(tasks)} images")
        if total_failed > 0:
            print(f"[WARNING] Total failed: {total_failed} images", file=sys.stderr)

        return temp_files

    def _load_and_merge_temp_files(self, temp_files: List[str]) -> List[Dict]:
        """Load and merge temporary files back into memory for shuffling."""
        print(f"  Loading {len(temp_files)} temporary files...")
        all_data = []

        for temp_file in temp_files:
            data = np.load(temp_file, allow_pickle=True)

            # Reconstruct dict format
            n_samples = len(data['images'])
            for i in range(n_samples):
                sample = {
                    'image': data['images'][i],
                    'path': str(data['image_paths'][i])
                }

                if 'bboxes' in data:
                    # Detector data (or teacher with old format)
                    if data['bboxes'][i].ndim == 1:
                        # Teacher data (single bbox)
                        sample['bbox'] = data['bboxes'][i]
                    else:
                        # Detector data (list of bboxes)
                        sample['bboxes'] = data['bboxes'][i]
                elif 'keypoints' in data:
                    # Landmarker data
                    sample['keypoints'] = data['keypoints'][i]

                all_data.append(sample)

            # Clean up temp file
            Path(temp_file).unlink()

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
                images=np.array([all_data[i]['image'] for i in train_indices],
                               dtype=object),
                bboxes=np.array([all_data[i]['bboxes'] for i in train_indices],
                               dtype=object),
                image_paths=np.array([all_data[i]['path'] for i in train_indices])
            )
        else:
            np.savez_compressed(
                train_file,
                images=np.array([all_data[i]['image'] for i in train_indices],
                               dtype=object),
                keypoints=np.array([all_data[i]['keypoints'] for i in train_indices],
                                  dtype=object),
                image_paths=np.array([all_data[i]['path'] for i in train_indices])
            )
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPZ
        val_file = self.output_dir / f'val_{data_type}.npz'
        if is_detector:
            np.savez_compressed(
                val_file,
                images=np.array([all_data[i]['image'] for i in val_indices],
                               dtype=object),
                bboxes=np.array([all_data[i]['bboxes'] for i in val_indices],
                               dtype=object),
                image_paths=np.array([all_data[i]['path'] for i in val_indices])
            )
        else:
            np.savez_compressed(
                val_file,
                images=np.array([all_data[i]['image'] for i in val_indices],
                               dtype=object),
                keypoints=np.array([all_data[i]['keypoints'] for i in val_indices],
                                  dtype=object),
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

        # Separate PTS and COCO tasks
        pts_tasks = [t for t in tasks if len(t) == 2]
        coco_tasks = [t for t in tasks if len(t) == 3]

        temp_files = []

        # Process PTS tasks
        if pts_tasks:
            print(f"  Processing {len(pts_tasks)} PTS images...")
            pts_files = self._process_tasks_in_batches(pts_tasks, process_pts_file,
                                                      'landmarker_pts')
            temp_files.extend(pts_files)

        # Process COCO landmarker tasks
        if coco_tasks:
            print(f"  Processing {len(coco_tasks)} COCO images...")
            coco_files = self._process_tasks_in_batches(coco_tasks, process_coco_landmarker_image,
                                                        'landmarker_coco')
            temp_files.extend(coco_files)

        # Load temp files, shuffle, and save final files
        all_data = self._load_and_merge_temp_files(temp_files)
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

    def process_teacher_data(self, train_split: float = 0.8) -> None:
        """
        Process teacher datasets (cropped ear images for autoencoder).
        Collects data from both detector and landmarker folders.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        print("Processing teacher data (cropped ears for autoencoder)...")
        print(f"Using {self.num_workers} workers for parallel processing")

        all_tasks = []

        # Collect from detector folder
        detector_dir = self.raw_data_dir / 'detector'
        if detector_dir.exists():
            print("  Collecting from detector datasets...")
            # COCO datasets
            coco_dirs = [d for d in detector_dir.glob('*')
                        if d.is_dir() and 'coco' in d.name.lower()]
            for coco_dir in coco_dirs:
                for split in ['train', 'test', 'valid']:
                    annotation_file = coco_dir / split / '_annotations.coco.json'
                    if annotation_file.exists():
                        all_tasks.extend(self._collect_coco_tasks(annotation_file,
                                                                  coco_dir / split))

            # CSV datasets
            csv_files = list(detector_dir.glob('**/*_annotations.csv'))
            for csv_file in csv_files:
                all_tasks.extend(self._collect_csv_tasks(csv_file, csv_file.parent))

        # Collect from landmarker folder
        landmarker_dir = self.raw_data_dir / 'landmarker'
        if landmarker_dir.exists():
            print("  Collecting from landmarker datasets...")
            # PTS datasets
            pts_collections = ['collectionA', 'collectionB']
            for collection in pts_collections:
                collection_dir = landmarker_dir / collection
                if collection_dir.exists():
                    all_tasks.extend(self._collect_pts_tasks(collection_dir))

            # COCO landmarker datasets
            coco_dirs = [d for d in landmarker_dir.glob('*')
                        if d.is_dir() and 'coco' in d.name.lower()]
            for coco_dir in coco_dirs:
                for split in ['train', 'test', 'valid']:
                    annotation_file = coco_dir / split / '_annotations.coco.json'
                    if annotation_file.exists():
                        all_tasks.extend(self._collect_coco_tasks(annotation_file,
                                                                  coco_dir / split))

        print(f"Found {len(all_tasks)} images to process")

        # Determine worker function based on task type
        if not all_tasks:
            print("No teacher data found!")
            return

        # Process different task types separately
        # Separate COCO, CSV, and PTS tasks
        coco_tasks = [t for t in all_tasks if len(t) == 3 and '.csv' not in t[0]]
        csv_tasks = [t for t in all_tasks if len(t) == 3 and '.csv' in t[0]]
        pts_tasks = [t for t in all_tasks if len(t) == 2]

        temp_files = []

        if coco_tasks:
            print(f"  Processing {len(coco_tasks)} COCO images...")
            coco_files = self._process_tasks_in_batches(coco_tasks,
                                                        process_teacher_image,
                                                        'teacher_coco')
            temp_files.extend(coco_files)

        if csv_tasks:
            print(f"  Processing {len(csv_tasks)} CSV images...")
            csv_files = self._process_tasks_in_batches(csv_tasks,
                                                       process_teacher_csv,
                                                       'teacher_csv')
            temp_files.extend(csv_files)

        if pts_tasks:
            print(f"  Processing {len(pts_tasks)} PTS images...")
            pts_files = self._process_tasks_in_batches(pts_tasks,
                                                       process_teacher_pts,
                                                       'teacher_pts')
            temp_files.extend(pts_files)

        if not temp_files:
            print("No valid teacher data processed!")
            return

        # Load temp files, shuffle, and save final files
        all_data = self._load_and_merge_temp_files(temp_files)

        # Split and save (teacher data only contains cropped images)
        n_samples = len(all_data)
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_split)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Save train NPZ
        train_file = self.output_dir / 'train_teacher.npz'
        np.savez_compressed(
            train_file,
            images=np.array([all_data[i]['image'] for i in train_indices],
                           dtype=object),
            image_paths=np.array([all_data[i]['path'] for i in train_indices]),
            bboxes=np.array([all_data[i]['bbox'] for i in train_indices],
                           dtype=object)
        )
        print(f"Saved {len(train_indices)} training samples to {train_file}")

        # Save validation NPZ
        val_file = self.output_dir / 'val_teacher.npz'
        np.savez_compressed(
            val_file,
            images=np.array([all_data[i]['image'] for i in val_indices],
                           dtype=object),
            image_paths=np.array([all_data[i]['path'] for i in val_indices]),
            bboxes=np.array([all_data[i]['bbox'] for i in val_indices],
                           dtype=object)
        )
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
        print("Data Processing Pipeline (Parallel & Memory-Efficient)")
        print("=" * 60)
        print(f"Batch size: {self.batch_size}")
        print(f"Workers: {self.num_workers}")
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
  python data_processor.py --all --batch-size 1000 --workers 8 --split 0.85
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
                           help='Process teacher data (cropped ears for autoencoder)')

    # Configuration arguments
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--batch-size', type=int, default=500,
                             help='Number of images to process per batch (default: 500)')
    config_group.add_argument('--workers', type=int, default=None,
                             help='Number of parallel workers (default: CPU count)')
    config_group.add_argument('--split', type=float, default=0.8,
                             help='Train/validation split ratio (default: 0.8)')
    config_group.add_argument('--input-dir', type=str, default='data/raw',
                             help='Input directory containing raw data (default: data/raw)')
    config_group.add_argument('--output-dir', type=str, default='data/preprocessed',
                             help='Output directory for NPZ files (default: data/preprocessed)')
    config_group.add_argument('--flush-every', type=int, default=5,
                             help='Flush data to disk every N batches (default: 5)')

    args = parser.parse_args()

    # Validate arguments
    if not (args.all or args.detector or args.landmarker or args.teacher):
        parser.error('Must specify at least one of: --all, --detector, --landmarker, --teacher')

    if args.split <= 0 or args.split >= 1:
        parser.error('--split must be between 0 and 1')

    # Create processor
    processor = DataProcessor(
        raw_data_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        flush_every=args.flush_every
    )

    # Determine what to process
    if args.all:
        # Process everything
        processor.process_all(train_split=args.split, include_teacher=True)
    else:
        # Process selected datasets
        print("=" * 60)
        print("Data Processing Pipeline (Parallel & Memory-Efficient)")
        print("=" * 60)
        print(f"Batch size: {processor.batch_size}")
        print(f"Workers: {processor.num_workers}")
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
