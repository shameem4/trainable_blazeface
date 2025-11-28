"""
Unified decoder that automatically selects the appropriate annotation decoder
based on available annotation files.
"""
import os
from coco_decoder import find_coco_annotation, decode_coco_annotation
from csv_decoder import find_csv_annotation, decode_csv_annotation
from pts_decoder import find_pts_annotation, decode_pts_annotation


def find_annotation(image_path):
    """
    Find corresponding annotation file for an image.
    Checks for COCO, CSV, and PTS formats in order.

    Args:
        image_path: Path to the image file

    Returns:
        tuple: (annotation_path, annotation_type) or (None, None) if not found
               annotation_type can be 'coco', 'csv', or 'pts'
    """
    coco_path = find_coco_annotation(image_path)
    csv_path = find_csv_annotation(image_path)
    pts_path = find_pts_annotation(image_path)

    if coco_path:
        return coco_path, 'coco'
    elif csv_path:
        return csv_path, 'csv'
    elif pts_path:
        return pts_path, 'pts'

    return None, None


def decode_annotation(annotation_path, image_path, annotation_type):
    """
    Decode annotation file using the appropriate decoder.

    Args:
        annotation_path: Path to the annotation file
        image_path: Path to the image file
        annotation_type: Type of annotation ('coco', 'csv', or 'pts')

    Returns:
        list: List of annotation dictionaries with 'bbox' and/or 'keypoints' keys
              Returns empty list if decoding fails or image not found
    """
    if annotation_type == 'coco':
        return decode_coco_annotation(annotation_path, image_path)
    elif annotation_type == 'csv':
        return decode_csv_annotation(annotation_path, image_path)
    elif annotation_type == 'pts':
        return decode_pts_annotation(annotation_path, image_path)
    else:
        print(f"Unknown annotation type: {annotation_type}")
        return []


def get_annotation_color(annotation_type):
    """
    Get the color to use for drawing annotations based on type.

    Args:
        annotation_type: Type of annotation ('coco', 'csv', or 'pts')

    Returns:
        str: Color name for drawing
    """
    color_map = {
        'coco': 'red',
        'csv': 'green',
        'pts': 'purple'
    }
    return color_map.get(annotation_type, 'blue')


def find_all_annotations(directory):
    """
    Find all annotation files in a directory and its subdirectories.

    Args:
        directory: Path to directory to search

    Returns:
        list: List of tuples (annotation_file_path, annotation_type, image_directory)
              annotation_type can be 'coco', 'csv', or 'pts'
    """
    from pathlib import Path

    directory = Path(directory)
    annotations = []

    # Find COCO annotation files
    for coco_file in directory.glob('**/_annotations.coco.json'):
        annotations.append((coco_file, 'coco', coco_file.parent))

    # Find CSV annotation files
    for csv_file in directory.glob('**/*_annotations.csv'):
        annotations.append((csv_file, 'csv', csv_file.parent))

    # Find PTS files (group by directory since they're per-image)
    pts_dirs = set()
    for pts_file in directory.glob('**/*.pts'):
        pts_dirs.add(pts_file.parent)

    for pts_dir in pts_dirs:
        annotations.append((None, 'pts', pts_dir))

    return annotations


def decode_all_annotations(annotation_file, annotation_type, image_dir, progress_callback=None):
    """
    Decode all annotations from a file or directory.

    Args:
        annotation_file: Path to annotation file (or None for PTS)
        annotation_type: Type of annotation ('coco', 'csv', or 'pts')
        image_dir: Directory containing images
        progress_callback: Optional callback function(current, total) for progress tracking

    Returns:
        list: List of dicts with keys:
              - 'image_path': str - Full path to image
              - 'bbox': list (for detector/teacher data)
              - 'keypoints': array (for landmarker data)

    Example:
        >>> annotations = decode_all_annotations('annotations.coco.json', 'coco', 'images/')
        >>> for ann in annotations:
        ...     print(f"Image: {ann['image_path']}, bbox: {ann.get('bbox')}")
    """
    import json
    import csv as csv_module
    from pathlib import Path

    annotation_file = Path(annotation_file) if annotation_file else None
    image_dir = Path(image_dir)
    results = []

    if annotation_type == 'coco':
        # COCO format - process all images in JSON
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        images = coco_data['images']
        total = len(images)

        for i, img_info in enumerate(images, 1):
            if progress_callback:
                progress_callback(i, total)

            image_path = image_dir / img_info['file_name']
            if not image_path.exists():
                continue

            try:
                annotations = decode_coco_annotation(str(annotation_file), img_info['file_name'])
                if annotations:
                    # Add image path to annotation
                    annotation = annotations[0].copy()
                    annotation['image_path'] = str(image_path)
                    results.append(annotation)
            except Exception:
                # Skip failed annotations silently (caller can log if needed)
                pass

    elif annotation_type == 'csv':
        # CSV format - process all unique images
        with open(annotation_file, 'r') as f:
            rows = list(csv_module.DictReader(f))

        # Get unique image names
        image_groups = {}
        for row in rows:
            img_name = row.get('image_path') or row.get('filename')
            if img_name and img_name not in image_groups:
                image_groups[img_name] = True

        image_names = list(image_groups.keys())
        total = len(image_names)

        for i, img_name in enumerate(image_names, 1):
            if progress_callback:
                progress_callback(i, total)

            image_path = image_dir / img_name
            if not image_path.exists():
                continue

            try:
                annotations = decode_csv_annotation(str(annotation_file), img_name)
                if annotations:
                    annotation = annotations[0].copy()
                    annotation['image_path'] = str(image_path)
                    results.append(annotation)
            except Exception:
                pass

    elif annotation_type == 'pts':
        # PTS format - process all .pts files in directory
        pts_files = list(image_dir.glob('**/*.pts'))
        total = len(pts_files)

        for i, pts_file in enumerate(pts_files, 1):
            if progress_callback:
                progress_callback(i, total)

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
                if annotations:
                    annotation = annotations[0].copy()
                    annotation['image_path'] = str(image_path)
                    results.append(annotation)
            except Exception:
                pass

    return results
