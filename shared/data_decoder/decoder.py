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
