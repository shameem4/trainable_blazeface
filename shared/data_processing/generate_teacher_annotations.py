"""
Script to generate COCO annotations for teacher dataset using YOLOv5 ear detection.
Processes all images in data/raw/teacher folder and creates _annotations.coco.json files.
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# Add YOLOv5 model path
yolo_path = Path(r'C:\Users\shame\OneDrive\Desktop\ear_stuff\yolo_ear\Automatic-measurement-of-human-ear-parameters-main')
sys.path.insert(0, str(yolo_path))

from nets.yolo import YoloBody
from utils.utils_bbox import DecodeBox


class EarDetector:
    """YOLOv5-based ear detector for generating bounding boxes."""

    def __init__(self, model_path=None, classes_path=None, anchors_path=None):
        """
        Initialize ear detector.

        Args:
            model_path: Path to YOLOv5 model weights
            classes_path: Path to classes file
            anchors_path: Path to anchors file
        """
        # Set default paths relative to yolo_path
        if model_path is None:
            model_path = yolo_path / 'yolo_image5.pth'
        if classes_path is None:
            classes_path = yolo_path / 'model_config' / 'ear_classes_image5.txt'
        if anchors_path is None:
            anchors_path = yolo_path / 'model_config' / 'yolo_anchors.txt'

        self.model_path = Path(model_path)
        self.classes_path = Path(classes_path)
        self.anchors_path = Path(anchors_path)

        # Model config
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.backbone = 'cspdarknet'
        self.phi = 's'
        self.input_shape = [640, 640]
        self.confidence = 0.4
        self.nms_iou = 0.3
        self.letterbox_image = True
        self.cuda = torch.cuda.is_available()

        # Load model
        self.class_names, self.num_classes = self._get_classes()
        self.anchors, self.num_anchors = self._get_anchors()
        self.bbox_util = DecodeBox(self.anchors, self.num_classes,
                                   (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)

        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi,
                           backbone=self.backbone, input_shape=self.input_shape)
        device = torch.device('cuda' if self.cuda else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()

        print(f"Loaded YOLOv5 model from {self.model_path}")
        print(f"Using device: {'CUDA' if self.cuda else 'CPU'}")

    def detect(self, image_path):
        """
        Detect ear bounding boxes in image.

        Args:
            image_path: Path to image file

        Returns:
            List of bounding boxes in format [x, y, width, height] or None if no detection
        """
        try:
            image = Image.open(image_path)
            image_shape = np.array(np.array(image).shape[0:2])

            # Preprocess
            image = self._cvt_color(image)
            image_data = self._resize_image(image, (self.input_shape[1], self.input_shape[0]),
                                           self.letterbox_image)
            image_data = np.expand_dims(
                np.transpose(self._preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

            # Run inference
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                if self.cuda:
                    images = images.cuda()

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)

                results = self.bbox_util.non_max_suppression(
                    torch.cat(outputs, 1), self.num_classes, self.input_shape,
                    image_shape, self.letterbox_image,
                    conf_thres=self.confidence, nms_thres=self.nms_iou)

                if results[0] is None or len(results[0]) == 0:
                    return None

                # Extract bboxes (top, left, bottom, right -> x, y, w, h)
                bboxes = []
                for detection in results[0]:
                    top, left, bottom, right = detection[:4]
                    x = float(left)
                    y = float(top)
                    w = float(right - left)
                    h = float(bottom - top)
                    bboxes.append([x, y, w, h])

                return bboxes

        except Exception as e:
            print(f"Error detecting ears in {image_path}: {e}")
            return None

    def _get_classes(self):
        """Load class names from file."""
        with open(self.classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    def _get_anchors(self):
        """Load anchors from file."""
        with open(self.anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors, len(anchors)

    @staticmethod
    def _cvt_color(image):
        """Convert image to RGB."""
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        else:
            return image.convert('RGB')

    @staticmethod
    def _resize_image(image, size, letterbox_image):
        """Resize image with letterboxing."""
        iw, ih = image.size
        w, h = size
        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    @staticmethod
    def _preprocess_input(image):
        """Normalize image."""
        image /= 255.0
        return image


def create_coco_annotation(image_dir, detector, output_path):
    """
    Create COCO format annotations for all images in a directory.

    Args:
        image_dir: Directory containing images
        detector: EarDetector instance
        output_path: Path to output JSON file
    """
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Initialize COCO structure
    coco_data = {
        'info': {
            'description': 'Ear detection dataset - auto-generated annotations',
            'version': '1.0',
            'year': datetime.now().year,
            'date_created': datetime.now().isoformat()
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {
                'id': 1,
                'name': 'ear',
                'supercategory': 'body'
            }
        ]
    }

    annotation_id = 1
    image_id = 1
    detected_count = 0

    print(f"Processing {len(image_files)} images in {image_dir.name}...")

    for image_path in tqdm(image_files, desc=f"Detecting ears in {image_dir.name}"):
        try:
            # Get image dimensions
            img = Image.open(image_path)
            width, height = img.size
            img.close()

            # Add image entry
            image_entry = {
                'id': image_id,
                'file_name': image_path.name,
                'width': width,
                'height': height
            }
            coco_data['images'].append(image_entry)

            # Detect ears
            bboxes = detector.detect(image_path)

            if bboxes:
                detected_count += 1
                for bbox in bboxes:
                    x, y, w, h = bbox

                    # Add annotation entry
                    annotation_entry = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,  # ear
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h),
                        'iscrowd': 0,
                        'segmentation': []
                    }
                    coco_data['annotations'].append(annotation_entry)
                    annotation_id += 1

            image_id += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    # Save COCO JSON
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Saved annotations to {output_path}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Images with detections: {detected_count}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")


def process_teacher_folder(teacher_dir='data/raw/teacher'):
    """
    Process all subdirectories in teacher folder and generate COCO annotations.

    Args:
        teacher_dir: Path to teacher data directory
    """
    teacher_dir = Path(teacher_dir)

    if not teacher_dir.exists():
        print(f"Error: {teacher_dir} does not exist!")
        return

    # Initialize detector
    print("Initializing YOLOv5 ear detector...")
    detector = EarDetector()

    # Find all subdirectories with images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    processed_dirs = []

    for subdir in teacher_dir.rglob('*'):
        if not subdir.is_dir():
            continue

        # Skip if already has COCO annotations
        if (subdir / '_annotations.coco.json').exists():
            print(f"Skipping {subdir.name} - annotations already exist")
            continue

        # Check if directory has images
        has_images = False
        for ext in image_extensions:
            if list(subdir.glob(f'*{ext}')) or list(subdir.glob(f'*{ext.upper()}')):
                has_images = True
                break

        if has_images:
            output_path = subdir / '_annotations.coco.json'
            create_coco_annotation(subdir, detector, output_path)
            processed_dirs.append(subdir.name)

    print(f"\nProcessing complete!")
    print(f"Generated annotations for {len(processed_dirs)} directories")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate COCO annotations for teacher dataset using YOLOv5')
    parser.add_argument('--teacher-dir', type=str, default='../../data/raw/teacher',
                       help='Path to teacher data directory')
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='Detection confidence threshold')

    args = parser.parse_args()

    # Run processing
    process_teacher_folder(args.teacher_dir)
