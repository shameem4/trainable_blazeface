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
- Modular bbox validation via BBoxChecker
- Parallel processing with configurable max cores
"""

# Add workspace root to path for standalone execution
import sys
from pathlib import Path
_workspace_root = Path(__file__).resolve().parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import sys
import traceback
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

# Import existing decoders
try:
    from shared.data_decoder.decoder import find_all_annotations, decode_all_annotations
except ImportError:
    import sys
    script_dir = Path(__file__).parent.resolve()
    decoder_dir = script_dir.parent / 'data_decoder'
    sys.path.insert(0, str(decoder_dir))
    from decoder import find_all_annotations, decode_all_annotations  # type: ignore

# Import bbox utilities
try:
    from shared.data_processing.bbox_utils import BBoxChecker, is_valid_bbox_xywh
except ImportError:
    from bbox_utils import BBoxChecker, is_valid_bbox_xywh

# Import anchor utilities for IoU-based filtering (optional, for detector data)
try:
    from ear_detector.anchors import (
        generate_anchors, 
        compute_iou, 
        anchors_to_xyxy, 
        MATCHING_CONFIG
    )
    ANCHOR_FILTERING_AVAILABLE = True
except ImportError:
    ANCHOR_FILTERING_AVAILABLE = False
    MATCHING_CONFIG = {'min_anchor_iou': 0.3}  # Fallback default

# Import YOLO detector for --detector-test option (optional)
try:
    from shared.data_processing.generate_teacher_annotations import EarDetector as YoloEarDetector
    YOLO_DETECTOR_AVAILABLE = True
except ImportError:
    YOLO_DETECTOR_AVAILABLE = False
    YoloEarDetector = None


# ============================================================================
# Parallel processing worker functions (must be at module level for pickling)
# ============================================================================

def _count_images_in_folder(ann_info: Tuple) -> Tuple[Tuple, int]:
    """
    Count the number of images in an annotation folder.
    
    Args:
        ann_info: Tuple of (annotation_file, format_type, image_dir)
        
    Returns:
        Tuple of (ann_info, image_count)
    """
    import json
    import csv as csv_module
    
    ann_file, format_type, image_dir = ann_info
    image_dir = Path(image_dir)
    
    count = 0
    try:
        if format_type == 'coco':
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            count = len(coco_data.get('images', []))
        elif format_type == 'csv':
            with open(ann_file, 'r') as f:
                rows = list(csv_module.DictReader(f))
            # Count unique images
            image_names = set()
            for row in rows:
                img_name = row.get('image_path') or row.get('filename')
                if img_name:
                    image_names.add(img_name)
            count = len(image_names)
        elif format_type == 'pts':
            count = len(list(image_dir.glob('*.pts')))
        elif format_type == 'images_only':
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            count = sum(1 for f in image_dir.iterdir() 
                       if f.suffix.lower() in image_extensions)
    except Exception:
        count = 0
    
    return (ann_info, count)


def _process_folder_chunk(
    ann_file: Optional[Path],
    format_type: str,
    image_dir: Path,
    data_type: str,
    start_idx: int,
    end_idx: int,
    chunk_id: int,
) -> Tuple[int, List[Dict], str]:
    """
    Process a chunk of images from a folder.
    
    This is the worker function that runs in parallel processes.
    
    Args:
        ann_file: Path to annotation file (or None)
        format_type: 'coco', 'csv', 'pts', or 'images_only'
        image_dir: Directory containing images
        data_type: 'detector', 'landmarker', or 'teacher'
        start_idx: Starting index for this chunk
        end_idx: Ending index for this chunk
        chunk_id: ID of this chunk for logging
        
    Returns:
        Tuple of (chunk_id, processed_data_list, error_message_or_empty)
    """
    import json
    import csv as csv_module
    from PIL import Image
    
    # Recreate BBoxChecker in worker process
    bbox_checker = BBoxChecker(min_width=0, min_height=0, allow_negative_coords=False)
    
    results = []
    error_msg = ""
    
    try:
        ann_file = Path(ann_file) if ann_file else None
        image_dir = Path(image_dir)
        
        # Get the list of items to process based on format
        items_to_process = []
        
        if format_type == 'coco':
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            items_to_process = coco_data['images'][start_idx:end_idx]
            
        elif format_type == 'csv':
            with open(ann_file, 'r') as f:
                rows = list(csv_module.DictReader(f))
            # Get unique images
            image_names = []
            seen = set()
            for row in rows:
                img_name = row.get('image_path') or row.get('filename')
                if img_name and img_name not in seen:
                    image_names.append(img_name)
                    seen.add(img_name)
            items_to_process = image_names[start_idx:end_idx]
            
        elif format_type == 'pts':
            pts_files = sorted(image_dir.glob('*.pts'))
            items_to_process = pts_files[start_idx:end_idx]
            
        elif format_type == 'images_only':
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = sorted([f for f in image_dir.iterdir() 
                                 if f.suffix.lower() in image_extensions])
            items_to_process = image_files[start_idx:end_idx]
        
        # Process each item
        for item in items_to_process:
            try:
                samples = _process_single_item(
                    item, ann_file, format_type, image_dir, data_type, bbox_checker
                )
                results.extend(samples)  # Add all samples (one per annotation)
            except Exception:
                # Skip failed items silently
                pass
                
    except Exception as e:
        error_msg = f"Chunk {chunk_id} error: {str(e)}"
    
    return (chunk_id, results, error_msg)


def _process_single_item(
    item,
    ann_file: Optional[Path],
    format_type: str,
    image_dir: Path,
    data_type: str,
    bbox_checker: BBoxChecker,
) -> List[Dict]:
    """
    Process a single item (image) and extract sample data.
    
    For detector data type, if an image has multiple annotations (e.g., multiple ears),
    this returns multiple samples - one per annotation.
    
    Args:
        item: The item to process (image info dict, image name, pts file, or image path)
        ann_file: Path to annotation file
        format_type: Annotation format type
        image_dir: Directory containing images
        data_type: Type of data to extract
        bbox_checker: BBoxChecker instance for validation
        
    Returns:
        List of sample dicts (may be empty if invalid)
    """
    from PIL import Image
    import json
    import csv as csv_module
    
    # Import decoders
    try:
        from shared.data_decoder.coco_decoder import decode_coco_annotation
        from shared.data_decoder.csv_decoder import decode_csv_annotation
        from shared.data_decoder.pts_decoder import decode_pts_annotation
    except ImportError:
        # Fallback for worker processes - these resolve at runtime
        script_dir = Path(__file__).parent.resolve()
        decoder_dir = script_dir.parent / 'data_decoder'
        sys.path.insert(0, str(decoder_dir))
        from coco_decoder import decode_coco_annotation  # type: ignore[import-not-found]
        from csv_decoder import decode_csv_annotation  # type: ignore[import-not-found]
        from pts_decoder import decode_pts_annotation  # type: ignore[import-not-found]
    
    annotations = []
    image_path = None
    
    if format_type == 'coco':
        # item is image info dict
        image_path = image_dir / item['file_name']
        if not image_path.exists():
            return []
        try:
            annotations = decode_coco_annotation(str(ann_file), item['file_name'])
        except Exception:
            return []
            
    elif format_type == 'csv':
        # item is image filename
        image_path = image_dir / item
        if not image_path.exists():
            return []
        try:
            annotations = decode_csv_annotation(str(ann_file), item)
        except Exception:
            return []
            
    elif format_type == 'pts':
        # item is pts file path
        image_path = item.with_suffix('.jpg')
        if not image_path.exists():
            image_path = item.with_suffix('.png')
        if not image_path.exists():
            return []
        try:
            annotations = decode_pts_annotation(str(item), str(image_path))
        except Exception:
            return []
            
    elif format_type == 'images_only':
        # item is image file path
        image_path = item
        annotations = [{}]  # Empty annotation, will use full image
    
    if image_path is None:
        return []
    
    # Extract sample data based on data_type
    # For detector, create one sample per annotation (handles multiple ears per image)
    return _extract_samples_from_annotations(
        annotations, str(image_path), data_type, bbox_checker
    )


def _extract_samples_from_annotations(
    annotations: List[Dict],
    image_path: str,
    data_type: str,
    bbox_checker: BBoxChecker,
) -> List[Dict]:
    """
    Extract sample data from a list of annotations.
    
    For detector data type, creates one sample per annotation (bbox).
    This allows images with multiple ears to appear multiple times in training.
    
    Args:
        annotations: List of annotation dicts from decoders
        image_path: Path to the image
        data_type: 'detector', 'landmarker', or 'teacher'
        bbox_checker: BBoxChecker instance for validation
        
    Returns:
        List of sample dicts
    """
    from PIL import Image
    
    if not annotations:
        annotations = [{}]
    
    samples = []
    
    if data_type == 'detector':
        # Create one sample per bbox (handles multiple ears per image)
        for annotation in annotations:
            if 'bbox' in annotation:
                bbox = annotation['bbox']
                if bbox_checker.is_valid_xywh(bbox):
                    samples.append({'path': image_path, 'bboxes': [bbox]})
        return samples
            
    elif data_type == 'landmarker':
        # Create one sample per annotation with keypoints
        for annotation in annotations:
            if 'keypoints' in annotation:
                try:
                    kpts = np.array(annotation['keypoints'])
                    if kpts.size == 0 or kpts.size % 3 != 0:
                        continue
                    kpts = kpts.reshape(-1, 3)
                    samples.append({'path': image_path, 'keypoints': kpts})
                except (ValueError, AttributeError):
                    continue
        return samples
            
    elif data_type == 'teacher':
        # Teacher: create one sample per annotation (bbox or keypoints-derived bbox)
        for annotation in annotations:
            if 'bbox' in annotation:
                bbox = annotation['bbox']
                if bbox_checker.is_valid_xywh(bbox):
                    samples.append({'path': image_path, 'bboxes': [bbox]})
            elif 'keypoints' in annotation:
                try:
                    kpts = np.array(annotation['keypoints'])
                    if kpts.size == 0 or kpts.size % 3 != 0:
                        continue
                    kpts = kpts.reshape(-1, 3)
                    x_coords, y_coords = kpts[:, 0], kpts[:, 1]
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()
                    padding_x = (x_max - x_min) * 0.1
                    padding_y = (y_max - y_min) * 0.1
                    bbox = [
                        max(0, x_min - padding_x),
                        max(0, y_min - padding_y),
                        (x_max - x_min) + 2 * padding_x,
                        (y_max - y_min) + 2 * padding_y
                    ]
                    if bbox_checker.is_valid_xywh(bbox):
                        samples.append({'path': image_path, 'bboxes': [bbox]})
                except (ValueError, AttributeError):
                    continue
        
        # If no annotations found, use full image as bbox
        if not samples:
            try:
                from PIL import Image
                img = Image.open(image_path)
                width, height = img.size
                img.close()
                samples.append({'path': image_path, 'bboxes': [[0, 0, width, height]]})
            except Exception:
                pass
        return samples
    
    return samples




class DataProcessor:
    """Processes raw data into preprocessed NPY metadata files for training."""

    def __init__(self, raw_data_dir: str = 'data/raw',
                 output_dir: str = 'data/preprocessed',
                 bbox_checker: BBoxChecker = None,
                 max_workers: int = 8,
                 images_per_worker: int = 1000):
        """
        Initialize data processor.

        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory for preprocessed output
            bbox_checker: Optional BBoxChecker instance for validation.
                         Uses default checker if not provided.
            max_workers: Maximum number of parallel workers (default: 8)
            images_per_worker: Number of images per worker for large folders (default: 1000)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.images_per_worker = images_per_worker
        
        # Use provided checker or create default
        self.bbox_checker = bbox_checker or BBoxChecker(
            min_width=0,
            min_height=0,
            allow_negative_coords=False
        )

    def _calculate_worker_allocation(
        self, 
        folder_counts: List[Tuple[Tuple, int]]
    ) -> List[Tuple[Tuple, int, List[Tuple[int, int]]]]:
        """
        Calculate how many workers each folder needs based on image count.
        
        Args:
            folder_counts: List of ((ann_file, format_type, image_dir), image_count)
            
        Returns:
            List of (ann_info, image_count, [(start_idx, end_idx), ...]) with chunk assignments
        """
        # Calculate workers needed per folder (1 per 1000 images, minimum 1)
        folder_workers = []
        total_workers_needed = 0
        
        for ann_info, count in folder_counts:
            if count == 0:
                continue
            workers_needed = max(1, (count + self.images_per_worker - 1) // self.images_per_worker)
            folder_workers.append((ann_info, count, workers_needed))
            total_workers_needed += workers_needed
        
        # Scale down if exceeding max_workers
        if total_workers_needed > self.max_workers:
            scale_factor = self.max_workers / total_workers_needed
            for i, (ann_info, count, workers) in enumerate(folder_workers):
                scaled_workers = max(1, int(workers * scale_factor))
                folder_workers[i] = (ann_info, count, scaled_workers)
        
        # Create chunk assignments for each folder
        result = []
        for ann_info, count, workers in folder_workers:
            chunk_size = (count + workers - 1) // workers
            chunks = []
            for w in range(workers):
                start_idx = w * chunk_size
                end_idx = min((w + 1) * chunk_size, count)
                if start_idx < count:
                    chunks.append((start_idx, end_idx))
            result.append((ann_info, count, chunks))
        
        return result

    def _process_data_type_parallel(self, directories: List[Path], data_type: str,
                                    train_split: float = 0.8,
                                    filter_by_anchor_iou: bool = False) -> None:
        """
        Process data type using parallel processing.

        Args:
            directories: List of directories to search for annotations
            data_type: Type of data to extract ('detector', 'landmarker', or 'teacher')
            train_split: Fraction of data to use for training
            filter_by_anchor_iou: If True and data_type='detector', filter samples
                by minimum anchor IoU coverage
        """
        print(f"Processing {data_type} data (parallel mode, max {self.max_workers} workers)...")
        all_data = []

        # Find all annotation folders
        all_annotations = []
        for directory in directories:
            if not directory.exists():
                continue
            annotations = find_all_annotations(directory)
            all_annotations.extend(annotations)
        
        if not all_annotations:
            print(f"No {data_type} annotation folders found!")
            return
        
        print(f"  Found {len(all_annotations)} annotation folders")
        
        # Count images in each folder (can be parallelized too for many folders)
        print("  Counting images in folders...")
        folder_counts = []
        for ann_info in all_annotations:
            _, count = _count_images_in_folder(ann_info)
            folder_counts.append((ann_info, count))
            ann_file, format_type, image_dir = ann_info
            label = self._get_folder_label(ann_file, format_type, image_dir)
            print(f"    {label}: {count} images")
        
        # Calculate worker allocation
        allocations = self._calculate_worker_allocation(folder_counts)
        
        total_chunks = sum(len(chunks) for _, _, chunks in allocations)
        print(f"  Distributing work across {total_chunks} chunks...")
        
        # Submit all chunks for parallel processing
        all_futures = []
        chunk_id = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for ann_info, count, chunks in allocations:
                ann_file, format_type, image_dir = ann_info
                
                for start_idx, end_idx in chunks:
                    future = executor.submit(
                        _process_folder_chunk,
                        str(ann_file) if ann_file else None,
                        format_type,
                        str(image_dir),
                        data_type,
                        start_idx,
                        end_idx,
                        chunk_id,
                    )
                    all_futures.append((future, ann_info, chunk_id))
                    chunk_id += 1
            
            # Collect results with progress
            completed = 0
            errors = []
            
            for future in as_completed([f[0] for f in all_futures]):
                completed += 1
                self._print_progress(completed, total_chunks)
                
                try:
                    cid, results, error_msg = future.result()
                    if error_msg:
                        errors.append(error_msg)
                    all_data.extend(results)
                except Exception as e:
                    errors.append(f"Future error: {str(e)}")
        
        print()  # New line after progress
        
        if errors:
            print(f"  Warnings: {len(errors)} chunk(s) had errors")
            for err in errors[:5]:  # Show first 5 errors
                print(f"    - {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        
        if not all_data:
            print(f"No {data_type} data found!")
            return

        print(f"  Total: {len(all_data)} {data_type} samples")

        # Split and save (unified for all data types)
        self._split_and_save(all_data, train_split, data_type, filter_by_anchor_iou)

    def _get_folder_label(self, ann_file, format_type, image_dir) -> str:
        """Get a friendly label for a folder."""
        if format_type == 'pts':
            return f"{Path(image_dir).name} PTS files"
        elif format_type == 'images_only':
            return f"{Path(image_dir).name} (images only)"
        else:
            parent_name = Path(image_dir).parent.name if Path(image_dir).parent.name != Path(image_dir).name else ""
            return f"{parent_name}/{Path(image_dir).name} {format_type.upper()}"

    def _process_data_type(self, directories: List[Path], data_type: str,
                           train_split: float = 0.8,
                           filter_by_anchor_iou: bool = False) -> None:
        """
        Unified processing method for any data type from multiple directories.
        Uses parallel processing for better performance.

        Args:
            directories: List of directories to search for annotations
            data_type: Type of data to extract ('detector', 'landmarker', or 'teacher')
            train_split: Fraction of data to use for training
            filter_by_anchor_iou: If True and data_type='detector', filter samples
                by minimum anchor IoU coverage
        """
        # Use parallel processing
        self._process_data_type_parallel(directories, data_type, train_split, filter_by_anchor_iou)

    def process_detector_data(self, train_split: float = 0.8,
                              filter_by_anchor_iou: bool = True) -> None:
        """
        Process detector datasets into NPY files.
        Automatically detects COCO, CSV, and PTS formats.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
            filter_by_anchor_iou: If True, filter out samples where best anchor
                IoU < MATCHING_CONFIG['min_anchor_iou']. Default True.
        """
        self._process_data_type(
            directories=[self.raw_data_dir / 'detector'],
            data_type='detector',
            train_split=train_split,
            filter_by_anchor_iou=filter_by_anchor_iou
        )

    def process_detector_test_data(self, image_dir: str = None,
                                   train_split: float = 0.8,
                                   filter_by_anchor_iou: bool = True) -> None:
        """
        Process detector TEST data using YOLO model to generate GT bboxes.
        
        Instead of reading annotation files, this uses a pre-trained YOLO ear
        detector to generate bounding boxes for each image. Useful for creating
        test datasets or when annotation files are not available.

        Args:
            image_dir: Directory containing test images. If None, uses data/raw/detector_test
            train_split: Fraction of data to use for training (rest for validation)
            filter_by_anchor_iou: If True, filter out samples where best anchor
                IoU < MATCHING_CONFIG['min_anchor_iou']. Default True.
        """
        if not YOLO_DETECTOR_AVAILABLE:
            print("ERROR: YOLO detector not available. Cannot process detector test data.")
            print("Make sure generate_teacher_annotations.py and YOLO model are accessible.")
            return
        
        if image_dir is None:
            image_dir = self.raw_data_dir / 'detector_test'
        else:
            image_dir = Path(image_dir)
        
        if not image_dir.exists():
            print(f"ERROR: Image directory not found: {image_dir}")
            return
        
        print(f"Processing detector test data from: {image_dir}")
        print("Using YOLO model to generate bounding boxes...")
        
        # Initialize YOLO detector
        try:
            detector = YoloEarDetector()
        except Exception as e:
            print(f"ERROR: Failed to initialize YOLO detector: {e}")
            return
        
        # Find all images (recursively)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'**/*{ext}'))
            image_files.extend(image_dir.glob(f'**/*{ext.upper()}'))
        image_files = sorted(set(image_files))  # Remove duplicates and sort
        
        print(f"Found {len(image_files)} images")
        
        # Process each image with YOLO
        all_data = []
        no_detection_count = 0
        
        for img_path in tqdm(image_files, desc="Detecting ears"):
            bboxes = detector.detect(str(img_path))
            
            if bboxes is None or len(bboxes) == 0:
                no_detection_count += 1
                continue
            
            # Create one sample per detected bbox
            for bbox in bboxes:
                if self.bbox_checker.is_valid_xywh(bbox):
                    all_data.append({
                        'path': str(img_path),
                        'bboxes': [bbox]  # [x, y, w, h]
                    })
        
        print(f"Detected ears in {len(image_files) - no_detection_count}/{len(image_files)} images")
        print(f"Total samples (including multi-ear): {len(all_data)}")
        
        if not all_data:
            print("No detections found!")
            return
        
        # Split and save with optional anchor filtering
        self._split_and_save(all_data, train_split, 'detector_test', filter_by_anchor_iou)


    def _process_annotations(self, ann_file: Path, image_dir: Path,
                            format_type: str, data_type: str) -> List[Dict]:
        """
        Process all annotations using the shared batch decoder.

        Args:
            ann_file: Annotation file path (or None for PTS which are per-image)
            image_dir: Directory containing images
            format_type: 'coco', 'csv', or 'pts'
            data_type: 'detector', 'landmarker', or 'teacher'

        Returns:
            List of processed data dicts
        """
        # Use shared decoder's batch processing with progress callback
        annotations = decode_all_annotations(
            ann_file,
            format_type,
            image_dir,
            progress_callback=self._print_progress
        )

        print()  # New line after progress bar

        # Extract relevant data for the data type
        data = []
        for annotation in annotations:
            sample = self._extract_sample_data(
                annotation,
                annotation['image_path'],
                data_type
            )
            if sample:
                data.append(sample)

        return data

    def _print_progress(self, current: int, total: int, bar_length: int = 40) -> None:
        """Print a progress bar to stdout."""
        percent = current / total
        filled = int(bar_length * percent)
        bar = '#' * filled + '-' * (bar_length - filled)
        print(f'\r    Progress: [{bar}] {current}/{total} ({percent*100:.1f}%)', end='', flush=True)

    def _is_valid_bbox(self, bbox) -> bool:
        """
        Validate that a bbox is valid using the modular BBoxChecker.

        Args:
            bbox: Bounding box [x, y, w, h]

        Returns:
            True if bbox is valid, False otherwise
        """
        return self.bbox_checker.is_valid_xywh(bbox)

    def _extract_sample_data(self, annotation: Dict, image_path: str, data_type: str) -> Dict:
        """
        Extract relevant data from annotation based on data type.

        Args:
            annotation: Decoded annotation dict (or None for teacher without annotations)
            image_path: Path to image file
            data_type: 'detector', 'landmarker', or 'teacher'

        Returns:
            Sample dict or None if annotation doesn't match data type
        """
        sample = {'path': image_path}

        if data_type == 'detector':
            # Detector needs bboxes
            if 'bbox' in annotation:
                bbox = annotation['bbox']
                # Validate bbox
                if not self._is_valid_bbox(bbox):
                    return None  # Invalid bbox
                sample['bboxes'] = [bbox]  # Wrap in list
            else:
                return None  # No bbox available

        elif data_type == 'landmarker':
            # Landmarker needs keypoints
            if 'keypoints' in annotation:
                try:
                    kpts = np.array(annotation['keypoints'])
                    # Check if keypoints are valid (multiple of 3)
                    if kpts.size == 0 or kpts.size % 3 != 0:
                        return None  # Invalid keypoints
                    kpts = kpts.reshape(-1, 3)
                    sample['keypoints'] = kpts
                except (ValueError, AttributeError):
                    return None  # Failed to reshape
            else:
                return None  # No keypoints available

        elif data_type == 'teacher':
            # Teacher uses same 'bboxes' format as detector for consistency
            if annotation and 'bbox' in annotation:
                bbox = annotation['bbox']
                # Validate bbox
                if not self._is_valid_bbox(bbox):
                    return None  # Invalid bbox
                sample['bboxes'] = [bbox]  # Wrap in list like detector
            elif annotation and 'keypoints' in annotation:
                try:
                    # Compute bbox from keypoints with 10% padding
                    kpts = np.array(annotation['keypoints'])
                    # Check if keypoints are valid (multiple of 3)
                    if kpts.size == 0 or kpts.size % 3 != 0:
                        return None  # Invalid keypoints
                    kpts = kpts.reshape(-1, 3)
                    x_coords, y_coords = kpts[:, 0], kpts[:, 1]
                    x_min, x_max = x_coords.min(), x_coords.max()
                    y_min, y_max = y_coords.min(), y_coords.max()
                    padding_x = (x_max - x_min) * 0.1
                    padding_y = (y_max - y_min) * 0.1
                    bbox = [
                        max(0, x_min - padding_x),
                        max(0, y_min - padding_y),
                        (x_max - x_min) + 2 * padding_x,
                        (y_max - y_min) + 2 * padding_y
                    ]
                    # Validate computed bbox
                    if not self._is_valid_bbox(bbox):
                        return None  # Invalid computed bbox
                    sample['bboxes'] = [bbox]  # Wrap in list like detector
                except (ValueError, AttributeError):
                    return None  # Failed to compute bbox from keypoints
            else:
                # No annotations - use full image size as bbox
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    width, height = img.size
                    img.close()
                    sample['bboxes'] = [[0, 0, width, height]]  # Wrap in list like detector
                except Exception:
                    return None  # Failed to read image

        return sample

    def _split_and_save(self, all_data: List[Dict], train_split: float,
                       data_type: str, filter_by_anchor_iou: bool = False) -> None:
        """Split data and save to NPY metadata files.
        
        Args:
            all_data: List of sample dicts
            train_split: Fraction for training
            data_type: 'detector', 'landmarker', or 'teacher'
            filter_by_anchor_iou: If True and data_type='detector', filter out
                samples where best anchor IoU < MATCHING_CONFIG['min_anchor_iou'].
                Uses ear_detector.anchors utilities (no code duplication).
        """
        # Apply anchor IoU filtering for detector data if requested
        if filter_by_anchor_iou and data_type == 'detector' and ANCHOR_FILTERING_AVAILABLE:
            all_data = self._filter_by_anchor_iou(all_data)
        
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

    def _filter_by_anchor_iou(self, all_data: List[Dict]) -> List[Dict]:
        """
        Filter detector samples by anchor IoU coverage.
        
        Removes samples where the best matching anchor has IoU < min_anchor_iou.
        This uses the same logic as ear_detector.dataset filtering but applies
        it during preprocessing to avoid runtime overhead.
        
        Args:
            all_data: List of detector sample dicts with 'bboxes' key
            
        Returns:
            Filtered list of samples
        """
        import torch
        from PIL import Image
        
        min_iou = MATCHING_CONFIG['min_anchor_iou']
        print(f"\nFiltering by anchor IoU (min_anchor_iou={min_iou})...")
        
        # Generate anchors once
        anchors = generate_anchors()
        anchors_xyxy = anchors_to_xyxy(anchors)
        
        filtered_data = []
        total_before = len(all_data)
        
        for sample in all_data:
            bbox_xywh = sample['bboxes'][0]  # [x, y, w, h] in pixels
            image_path = sample['path']
            
            try:
                # Get image dimensions to normalize bbox
                img = Image.open(image_path)
                img_w, img_h = img.size
                img.close()
                
                # Normalize bbox to [x1, y1, x2, y2] in [0, 1]
                x, y, w, h = bbox_xywh
                x1 = x / img_w
                y1 = y / img_h
                x2 = (x + w) / img_w
                y2 = (y + h) / img_h
                
                # Clamp to [0, 1]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)
                
                # Compute IoU with anchors
                gt_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
                ious = compute_iou(anchors_xyxy, gt_box)  # (num_anchors, 1)
                best_iou = ious.max().item()
                
                if best_iou >= min_iou:
                    filtered_data.append(sample)
                    
            except Exception:
                # Skip samples that fail (can't read image, etc.)
                pass
        
        total_after = len(filtered_data)
        filtered_count = total_before - total_after
        filter_rate = filtered_count / total_before * 100 if total_before > 0 else 0
        print(f"  Filtered {filtered_count}/{total_before} samples ({filter_rate:.1f}%)")
        print(f"  Kept {total_after} samples with anchor IoU >= {min_iou}")
        
        return filtered_data

    def process_landmarker_data(self, train_split: float = 0.8) -> None:
        """
        Process landmarker datasets into NPY files.
        Automatically detects PTS, COCO, and CSV formats.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        self._process_data_type(
            directories=[self.raw_data_dir / 'landmarker'],
            data_type='landmarker',
            train_split=train_split
        )

    def process_teacher_data(self, train_split: float = 0.8) -> None:
        """
        Process teacher datasets from teacher folder only.
        Uses annotations if present, otherwise uses full image as bbox.

        Args:
            train_split: Fraction of data to use for training (rest for validation)
        """
        self._process_data_type(
            directories=[
                self.raw_data_dir / 'teacher'
            ],
            data_type='teacher',
            train_split=train_split
        )

    def process_all(self, train_split: float = 0.8,
                   include_teacher: bool = True,
                   filter_by_anchor_iou: bool = True) -> None:
        """
        Process all datasets (detector, landmarker, and teacher).

        Args:
            train_split: Fraction of data to use for training
            include_teacher: Whether to process teacher data for autoencoder
            filter_by_anchor_iou: If True, filter detector samples by anchor IoU
        """
        print("=" * 60)
        print("Data Processing Pipeline (Metadata-Only)")
        print("=" * 60)
        print()

        errors = []

        # Process detector data
        try:
            self.process_detector_data(train_split, filter_by_anchor_iou)
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
    data_group.add_argument('--detector-test', type=str, nargs='?', const='data/raw/detector',
                           metavar='IMAGE_DIR',
                           help='Process detector TEST data using YOLO model to generate GT bboxes. '
                                'Optionally specify image directory (default: data/raw/detector)')
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
    
    # Parallel processing arguments
    parallel_group = parser.add_argument_group('Parallel Processing')
    parallel_group.add_argument('--max-workers', type=int, default=8,
                               help='Maximum number of parallel workers (default: 8)')
    parallel_group.add_argument('--images-per-worker', type=int, default=1000,
                               help='Number of images per worker for large folders (default: 1000)')
    
    # Filtering arguments
    filter_group = parser.add_argument_group('Filtering')
    filter_group.add_argument('--no-anchor-filter', action='store_true',
                             help='Disable anchor IoU filtering for detector data. '
                                  'By default, samples with best anchor IoU < min_anchor_iou are filtered out.')

    args = parser.parse_args()

    # Validate arguments
    if not (args.all or args.detector or args.detector_test or args.landmarker or args.teacher):
        parser.error('Must specify at least one of: --all, --detector, --detector-test, --landmarker, --teacher')

    if args.split <= 0 or args.split >= 1:
        parser.error('--split must be between 0 and 1')

    # Create processor
    processor = DataProcessor(
        raw_data_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        images_per_worker=args.images_per_worker
    )

    # Determine what to process
    if args.all:
        # Process everything
        processor.process_all(
            train_split=args.split,
            include_teacher=True,
            filter_by_anchor_iou=not args.no_anchor_filter
        )
    else:
        # Process selected datasets
        print("=" * 60)
        print("Data Processing Pipeline (Metadata-Only)")
        print("=" * 60)
        print()

        errors = []

        if args.detector:
            try:
                processor.process_detector_data(
                    args.split,
                    filter_by_anchor_iou=not args.no_anchor_filter
                )
            except Exception as e:
                error_msg = f"Detector processing failed: {type(e).__name__}: {e}"
                print(f"\n[ERROR] {error_msg}", file=sys.stderr)
                traceback.print_exc()
                errors.append(error_msg)
            print()

        if args.detector_test:
            try:
                processor.process_detector_test_data(
                    image_dir=args.detector_test,
                    train_split=args.split,
                    filter_by_anchor_iou=not args.no_anchor_filter
                )
            except Exception as e:
                error_msg = f"Detector test processing failed: {type(e).__name__}: {e}"
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
