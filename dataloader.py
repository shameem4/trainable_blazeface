"""
Data loading utilities for ear detection and landmark models.

Following vincent1bt/blazeface-tensorflow methodology:
- Detector training uses anchor-based target encoding
- Boxes are matched to anchors by IoU
- Hard negative mining with 3:1 ratio
- Data augmentation: horizontal flip, brightness, saturation

Provides Dataset classes for:
- Detector training (images + bounding boxes → anchor targets)
- Landmarker training (images + keypoints)
- Teacher training (images + boxes + landmarks)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
import cv2


# =============================================================================
# Anchor Configuration (matching blazebase.py)
# =============================================================================

def generate_anchor_centers(input_size: int = 128) -> np.ndarray:
    """
    Generate anchor centers for BlazeFace detector.
    
    Creates 896 anchor centers:
    - 512 from 16x16 grid (2 anchors per cell)
    - 384 from 8x8 grid (6 anchors per cell)
    
    Returns:
        anchors: (896, 2) array of [x, y] centers normalized to [0, 1]
    """
    # Small anchors: 16x16 grid, 2 anchors per cell
    small_coords = np.linspace(0.03125, 0.96875, 16, dtype=np.float32)
    small_x = np.tile(np.repeat(small_coords, 2), 16)  # (512,)
    small_y = np.repeat(small_coords, 32)  # (512,)
    small = np.stack([small_x, small_y], axis=1)  # (512, 2)
    
    # Big anchors: 8x8 grid, 6 anchors per cell
    big_coords = np.linspace(0.0625, 0.9375, 8, dtype=np.float32)
    big_x = np.tile(np.repeat(big_coords, 6), 8)  # (384,)
    big_y = np.repeat(big_coords, 48)  # (384,)
    big = np.stack([big_x, big_y], axis=1)  # (384, 2)
    
    return np.concatenate([small, big], axis=0)  # (896, 2)


def compute_iou(box: np.ndarray, anchor_box: np.ndarray) -> float:
    """
    Compute IoU between a box and anchor box.
    
    Following vincent1bt's implementation.
    
    Args:
        box: [ymin, xmin, ymax, xmax] ground truth box (MediaPipe convention)
        anchor_box: [ymin, xmin, ymax, xmax] anchor box (MediaPipe convention)
        
    Returns:
        IoU value
    """
    # MediaPipe convention: [ymin, xmin, ymax, xmax]
    y_min = max(box[0], anchor_box[0])
    x_min = max(box[1], anchor_box[1])
    y_max = min(box[2], anchor_box[2])
    x_max = min(box[3], anchor_box[3])
    
    overlap_area = max(0.0, x_max - x_min + 1) * max(0.0, y_max - y_min + 1)
    
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    anchor_area = (anchor_box[2] - anchor_box[0] + 1) * (anchor_box[3] - anchor_box[1] + 1)
    
    union_area = float(box_area + anchor_area - overlap_area)
    
    return overlap_area / union_area if union_area > 0 else 0.0


def encode_boxes_to_anchors(
    boxes: np.ndarray,
    input_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode ground truth boxes to anchor targets.
    
    Following vincent1bt's create_boxes() function:
    - Find best matching anchor for each box by IoU
    - Return targets for small (16x16) and big (8x8) grids
    
    Args:
        boxes: (N, 4) array of [ymin, xmin, ymax, xmax] normalized boxes (MediaPipe convention)
        input_size: Input image size (for IoU computation)
        
    Returns:
        small_anchors: (16, 16, 5) array [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
        big_anchors: (8, 8, 5) array [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
    """
    # Anchor sizes (normalized)
    small_size = 0.03125  # 1/32 = 4 pixels at 128
    big_size = 0.0625     # 1/16 = 8 pixels at 128
    
    # Anchor coordinates
    small_coords = np.linspace(0.03125, 0.96875, 16, dtype=np.float32)
    big_coords = np.linspace(0.0625, 0.9375, 8, dtype=np.float32)
    
    # Initialize targets
    small_anchor = np.zeros((16, 16, 5), dtype=np.float32)
    big_anchor = np.zeros((8, 8, 5), dtype=np.float32)
    
    for box in boxes:
        # MediaPipe convention: [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = box
        
        # Try small anchors (16x16 grid)
        best_iou_small = 0.01
        best_idx_small = None
        
        for y_idx, y_coord in enumerate(small_coords):
            for x_idx, x_coord in enumerate(small_coords):
                # Anchor box centered at (x_coord, y_coord)
                # In MediaPipe convention: [ymin, xmin, ymax, xmax]
                aymin = y_coord - small_size
                axmin = x_coord - small_size
                aymax = y_coord + small_size
                axmax = x_coord + small_size
                
                iou = compute_iou(
                    box * input_size, 
                    np.array([aymin, axmin, aymax, axmax]) * input_size
                )
                
                if iou > best_iou_small:
                    best_iou_small = iou
                    best_idx_small = (y_idx, x_idx)
        
        if best_idx_small is not None:
            y_idx, x_idx = best_idx_small
            small_anchor[y_idx, x_idx] = [1.0, ymin, xmin, ymax, xmax]
        
        # Try big anchors (8x8 grid)
        best_iou_big = 0.01
        best_idx_big = None
        
        for y_idx, y_coord in enumerate(big_coords):
            for x_idx, x_coord in enumerate(big_coords):
                # In MediaPipe convention: [ymin, xmin, ymax, xmax]
                aymin = y_coord - big_size
                axmin = x_coord - big_size
                aymax = y_coord + big_size
                axmax = x_coord + big_size
                
                iou = compute_iou(
                    box * input_size,
                    np.array([aymin, axmin, aymax, axmax]) * input_size
                )
                
                if iou > best_iou_big:
                    best_iou_big = iou
                    best_idx_big = (y_idx, x_idx)
        
        if best_idx_big is not None:
            y_idx, x_idx = best_idx_big
            big_anchor[y_idx, x_idx] = [1.0, ymin, xmin, ymax, xmax]
    
    return small_anchor, big_anchor


def flatten_anchor_targets(
    small_anchors: np.ndarray,
    big_anchors: np.ndarray
) -> np.ndarray:
    """Flatten anchor targets to (896, 5) format.
    
    Args:
        small_anchors: (16, 16, 5)
        big_anchors: (8, 8, 5)
        
    Returns:
        targets: (896, 5) array [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
    """
    small_flat = small_anchors.reshape(-1, 5)  # (256, 5) -> but we need 512
    big_flat = big_anchors.reshape(-1, 5)      # (64, 5) -> but we need 384
    
    # Replicate to match anchor count (2 per cell for small, 6 per cell for big)
    small_expanded = np.repeat(small_flat, 2, axis=0)  # (512, 5)
    big_expanded = np.repeat(big_flat, 6, axis=0)      # (384, 5)
    
    return np.concatenate([small_expanded, big_expanded], axis=0)  # (896, 5)


class BaseEarDataset(Dataset):
    """Base dataset class for ear detection/landmark models."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        """
        Args:
            npy_path: Path to NPY file with annotations
            root_dir: Optional root directory for image paths
            transform: Optional transform to apply to images
            target_size: Target image size (height, width)
        """
        self.npy_path = Path(npy_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        self.target_size = target_size
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load and parse NPY metadata file."""
        metadata = np.load(self.npy_path, allow_pickle=True).item()
        
        self.image_paths = metadata['image_paths']
        self.metadata = metadata
        
        # Determine data type
        self._parse_annotations(metadata)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse annotations from metadata. Override in subclasses."""
        raise NotImplementedError
    
    def _get_image_path(self, idx: int) -> Path:
        """Get absolute image path."""
        image_path = Path(self.image_paths[idx])
        
        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path
        
        return image_path
    
    def _load_image(self, idx: int) -> np.ndarray:
        """Load and preprocess image."""
        image_path = self._get_image_path(idx)
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _resize_image(
        self,
        image: np.ndarray,
        annotations: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize image and scale annotations.
        
        Args:
            image: Original image
            annotations: Dict with bboxes/keypoints
            
        Returns:
            Resized image and scaled annotations
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Resize image
        resized = cv2.resize(image, (target_w, target_h))
        
        # Scale factors
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Scale annotations
        scaled_annotations = {}
        
        if 'bboxes' in annotations:
            bboxes = np.array(annotations['bboxes'])
            if len(bboxes) > 0:
                # Scale x, y, w, h
                bboxes[:, 0] *= scale_x  # x
                bboxes[:, 1] *= scale_y  # y
                bboxes[:, 2] *= scale_x  # w
                bboxes[:, 3] *= scale_y  # h
            scaled_annotations['bboxes'] = bboxes
        
        if 'keypoints' in annotations:
            keypoints = np.array(annotations['keypoints'])
            if len(keypoints) > 0:
                # Scale x, y coordinates
                keypoints[:, 0::2] *= scale_x  # x coords
                keypoints[:, 1::2] *= scale_y  # y coords
            scaled_annotations['keypoints'] = keypoints
        
        return resized, scaled_annotations
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class DetectorDataset(BaseEarDataset):
    """
    Dataset for detector training (images + bounding boxes).
    
    Following vincent1bt/blazeface-tensorflow:
    - Encodes boxes to anchor targets (896 anchors)
    - Returns flattened targets: (896, 5) with [class, x1, y1, x2, y2]
    - Supports data augmentation: horizontal flip, brightness, saturation
    """
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
        augment: bool = True
    ):
        """
        Args:
            npy_path: Path to NPY file
            root_dir: Root directory for images
            transform: Optional image transform
            target_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.augment = augment
        self.anchor_centers = generate_anchor_centers(target_size[0])
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse detector annotations."""
        self.bboxes = metadata.get('bboxes', [])
    
    def _random_crop_scale_translate(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translate_range: float = 0.2,
        min_box_visibility: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random crop, scale, and translation to move targets away from center.
        
        This prevents the model from learning a center bias.
        
        Args:
            image: RGB image (H, W, 3)
            bboxes: (N, 4) boxes in [ymin, xmin, ymax, xmax] normalized format (MediaPipe convention)
            scale_range: Min/max scale factors
            translate_range: Max translation as fraction of image size
            min_box_visibility: Minimum fraction of box that must remain visible
            
        Returns:
            Cropped/transformed image, adjusted boxes
        """
        h, w = image.shape[:2]
        
        # Random scale
        scale = np.random.uniform(scale_range[0], scale_range[1])
        
        # Random translation (in normalized coordinates)
        tx = np.random.uniform(-translate_range, translate_range)
        ty = np.random.uniform(-translate_range, translate_range)
        
        # Calculate crop region in original image coordinates
        # When scale > 1, we zoom in (crop smaller region)
        # When scale < 1, we zoom out (need padding)
        
        crop_w = w / scale
        crop_h = h / scale
        
        # Center of crop with translation offset
        cx = w / 2 + tx * w
        cy = h / 2 + ty * h
        
        # Crop bounds
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = int(cx + crop_w / 2)
        y2 = int(cy + crop_h / 2)
        
        # Handle out-of-bounds with padding
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        
        # Clamp to image bounds
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(w, x2)
        y2_clamped = min(h, y2)
        
        # Extract crop region
        cropped = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        
        # Add padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                int(pad_top), int(pad_bottom),
                int(pad_left), int(pad_right),
                cv2.BORDER_CONSTANT,
                value=(128, 128, 128)  # Gray padding
            )
        
        # Resize back to original size
        result_image = cv2.resize(cropped, (w, h))
        
        # Adjust bounding boxes - MediaPipe convention [ymin, xmin, ymax, xmax]
        if len(bboxes) == 0:
            return result_image, bboxes
        
        # Transform boxes to new coordinate system
        # Original box coords are in [0, 1] normalized space
        # We need to map them to the crop region
        
        # Convert normalized coords to pixel coords
        # MediaPipe: [ymin, xmin, ymax, xmax]
        boxes_pixel = bboxes.copy()
        boxes_pixel[:, [1, 3]] *= w  # xmin, xmax
        boxes_pixel[:, [0, 2]] *= h  # ymin, ymax
        
        # Shift by crop offset (accounting for padding)
        boxes_pixel[:, 1] -= (x1 - pad_left)  # xmin
        boxes_pixel[:, 3] -= (x1 - pad_left)  # xmax
        boxes_pixel[:, 0] -= (y1 - pad_top)   # ymin
        boxes_pixel[:, 2] -= (y1 - pad_top)   # ymax
        
        # Scale by crop size ratio
        actual_crop_w = crop_w + pad_left + pad_right - (x1_clamped - x1) - (x2 - x2_clamped)
        actual_crop_h = crop_h + pad_top + pad_bottom - (y1_clamped - y1) - (y2 - y2_clamped)
        
        # The crop region spans from (x1-pad_left) to (x2+pad_right) in the padded space
        total_crop_w = int(crop_w) + pad_left + pad_right
        total_crop_h = int(crop_h) + pad_top + pad_bottom
        
        if total_crop_w > 0 and total_crop_h > 0:
            boxes_pixel[:, [1, 3]] *= w / total_crop_w  # xmin, xmax
            boxes_pixel[:, [0, 2]] *= h / total_crop_h  # ymin, ymax
        
        # Clamp to image bounds
        boxes_pixel[:, 1] = np.clip(boxes_pixel[:, 1], 0, w)  # xmin
        boxes_pixel[:, 3] = np.clip(boxes_pixel[:, 3], 0, w)  # xmax
        boxes_pixel[:, 0] = np.clip(boxes_pixel[:, 0], 0, h)  # ymin
        boxes_pixel[:, 2] = np.clip(boxes_pixel[:, 2], 0, h)  # ymax
        
        # Convert back to normalized coords
        new_bboxes = boxes_pixel.copy()
        new_bboxes[:, [1, 3]] /= w  # xmin, xmax
        new_bboxes[:, [0, 2]] /= h  # ymin, ymax
        
        # Filter out boxes that are too small (mostly cropped out)
        # Area = (xmax - xmin) * (ymax - ymin)
        valid_boxes = []
        for i, (old_box, new_box) in enumerate(zip(bboxes, new_bboxes)):
            old_area = (old_box[3] - old_box[1]) * (old_box[2] - old_box[0])
            new_area = (new_box[3] - new_box[1]) * (new_box[2] - new_box[0])
            
            # Keep box if enough is still visible
            if old_area > 0 and (new_area / old_area) >= min_box_visibility:
                # Also check minimum size (width and height)
                if (new_box[3] - new_box[1]) > 0.02 and (new_box[2] - new_box[0]) > 0.02:
                    valid_boxes.append(new_box)
        
        if len(valid_boxes) > 0:
            return result_image, np.array(valid_boxes)
        else:
            # If all boxes were cropped out, return original
            return image, bboxes
    
    def _apply_synthetic_occlusion(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        max_occluders: int = 3,
        occluder_size_range: Tuple[float, float] = (0.05, 0.3),
        occlusion_types: List[str] = ['rectangle', 'ellipse', 'random_noise']
    ) -> np.ndarray:
        """
        Apply synthetic occlusion to simulate real-world scenarios.
        
        Simulates:
        - Hair covering ears
        - Hands/objects blocking view
        - Headphones, earrings, etc.
        - Random noise/artifacts
        
        Args:
            image: RGB image (H, W, 3)
            bboxes: (N, 4) boxes in [ymin, xmin, ymax, xmax] normalized format (MediaPipe convention)
            max_occluders: Maximum number of occluders to add
            occluder_size_range: Min/max size as fraction of image
            occlusion_types: Types of occlusion to apply
            
        Returns:
            Image with synthetic occlusions
        """
        h, w = image.shape[:2]
        image = image.copy()
        
        num_occluders = np.random.randint(1, max_occluders + 1)
        
        for _ in range(num_occluders):
            occluder_type = np.random.choice(occlusion_types)
            
            # Random size
            size_frac = np.random.uniform(occluder_size_range[0], occluder_size_range[1])
            occ_w = int(w * size_frac * np.random.uniform(0.5, 1.5))
            occ_h = int(h * size_frac * np.random.uniform(0.5, 1.5))
            
            # Bias position towards bounding boxes (70% chance to overlap)
            if len(bboxes) > 0 and np.random.random() > 0.3:
                # Pick a random box to occlude
                box_idx = np.random.randint(len(bboxes))
                box = bboxes[box_idx]  # [ymin, xmin, ymax, xmax]
                
                # Position occluder near/over the box - MediaPipe convention
                box_cx = (box[1] + box[3]) / 2 * w  # (xmin + xmax) / 2
                box_cy = (box[0] + box[2]) / 2 * h  # (ymin + ymax) / 2
                
                # Random offset from box center
                offset_x = np.random.uniform(-0.5, 0.5) * (box[3] - box[1]) * w  # width = xmax - xmin
                offset_y = np.random.uniform(-0.5, 0.5) * (box[2] - box[0]) * h  # height = ymax - ymin
                
                occ_x = int(box_cx + offset_x - occ_w / 2)
                occ_y = int(box_cy + offset_y - occ_h / 2)
            else:
                # Random position anywhere
                occ_x = np.random.randint(-occ_w // 2, w - occ_w // 2)
                occ_y = np.random.randint(-occ_h // 2, h - occ_h // 2)
            
            # Clamp to image bounds
            x1 = max(0, occ_x)
            y1 = max(0, occ_y)
            x2 = min(w, occ_x + occ_w)
            y2 = min(h, occ_y + occ_h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            if occluder_type == 'rectangle':
                # Solid or semi-transparent rectangle
                color = tuple(np.random.randint(0, 256, 3).tolist())
                alpha = np.random.uniform(0.3, 1.0)
                
                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
            elif occluder_type == 'ellipse':
                # Ellipse (simulates hand, round objects)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                alpha = np.random.uniform(0.3, 1.0)
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                angle = np.random.randint(0, 180)
                
                overlay = image.copy()
                cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
                
            elif occluder_type == 'random_noise':
                # Random noise patch (simulates artifacts, blur)
                noise = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
                alpha = np.random.uniform(0.2, 0.6)
                
                roi = image[y1:y2, x1:x2]
                blended = cv2.addWeighted(noise, alpha, roi, 1 - alpha, 0)
                image[y1:y2, x1:x2] = blended
                
            elif occluder_type == 'blur':
                # Gaussian blur patch
                kernel_size = np.random.choice([5, 7, 9, 11, 15])
                roi = image[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                image[y1:y2, x1:x2] = blurred
                
            elif occluder_type == 'cutout':
                # Black or gray cutout (like CutOut augmentation)
                gray_value = np.random.randint(0, 128)
                image[y1:y2, x1:x2] = gray_value
        
        return image
    
    def _augment_image(
        self,
        image: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Apply data augmentation following vincent1bt + random crop/scale/translate.
        
        Args:
            image: RGB image (H, W, 3)
            bboxes: (N, 4) boxes in [ymin, xmin, ymax, xmax] normalized format (MediaPipe convention)
            
        Returns:
            Augmented image, boxes, and whether horizontal flip was applied
        """
        horizontal_flip = False
        
        if not self.augment:
            return image, bboxes, horizontal_flip
        
        # Random crop/scale/translate (70% chance) - moves targets away from center
        if np.random.random() > 0.3:
            image, bboxes = self._random_crop_scale_translate(
                image, bboxes,
                scale_range=(0.8, 1.3),
                translate_range=0.25
            )
        
        # Random saturation (50% chance)
        if np.random.random() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            saturation_factor = np.random.uniform(0.5, 1.5)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random brightness (50% chance)
        if np.random.random() > 0.5:
            brightness_delta = np.random.uniform(-0.2, 0.2) * 255
            image = np.clip(image.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)
        
        # Synthetic occlusion (40% chance) - simulate hair, hands, objects
        if np.random.random() > 0.6:
            image = self._apply_synthetic_occlusion(
                image, bboxes,
                max_occluders=3,
                occluder_size_range=(0.05, 0.25),
                occlusion_types=['rectangle', 'ellipse', 'random_noise', 'cutout']
            )
        
        # Random horizontal flip (50% chance)
        if np.random.random() > 0.5:
            horizontal_flip = True
            image = np.fliplr(image).copy()
            
            # Flip box coordinates - MediaPipe convention [ymin, xmin, ymax, xmax]
            # Only x coords change: xmin_new = 1 - xmax_old, xmax_new = 1 - xmin_old
            if len(bboxes) > 0:
                xmin_old = bboxes[:, 1].copy()
                xmax_old = bboxes[:, 3].copy()
                bboxes[:, 1] = 1.0 - xmax_old  # xmin
                bboxes[:, 3] = 1.0 - xmin_old  # xmax
        
        return image, bboxes, horizontal_flip
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor normalized to [0, 1]
                - anchor_targets: [896, 5] tensor [class, ymin, xmin, ymax, xmax] (MediaPipe convention)
                - small_anchors: [16, 16, 5] for visualization/debugging
                - big_anchors: [8, 8, 5] for visualization/debugging
        """
        # Load image
        image = self._load_image(idx)
        
        # Get bboxes
        raw_bboxes = np.array(self.bboxes[idx]) if self.bboxes else np.array([])
        
        # Convert from [x, y, w, h] to MediaPipe convention [ymin, xmin, ymax, xmax]
        if len(raw_bboxes) > 0:
            if raw_bboxes.shape[1] == 4:
                # Assume [x, y, w, h] format, convert to [ymin, xmin, ymax, xmax]
                bboxes = np.zeros_like(raw_bboxes)
                bboxes[:, 0] = raw_bboxes[:, 1]  # ymin = y
                bboxes[:, 1] = raw_bboxes[:, 0]  # xmin = x
                bboxes[:, 2] = raw_bboxes[:, 1] + raw_bboxes[:, 3]  # ymax = y + h
                bboxes[:, 3] = raw_bboxes[:, 0] + raw_bboxes[:, 2]  # xmax = x + w
            else:
                bboxes = raw_bboxes
        else:
            bboxes = np.array([]).reshape(0, 4)
        
        # Resize image
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.target_size
        image = cv2.resize(image, (target_w, target_h))
        
        # Scale bboxes to [0, 1] range - MediaPipe convention [ymin, xmin, ymax, xmax]
        if len(bboxes) > 0:
            bboxes[:, 0] /= orig_h  # ymin
            bboxes[:, 1] /= orig_w  # xmin
            bboxes[:, 2] /= orig_h  # ymax
            bboxes[:, 3] /= orig_w  # xmax
            bboxes = np.clip(bboxes, 0, 1)
        
        # Apply augmentation
        image, bboxes, _ = self._augment_image(image, bboxes)
        
        # Encode boxes to anchor targets
        small_anchors, big_anchors = encode_boxes_to_anchors(
            bboxes, input_size=self.target_size[0]
        )
        
        # Flatten to (896, 5)
        anchor_targets = flatten_anchor_targets(small_anchors, big_anchors)
        
        # Apply custom transform or default normalization
        if self.transform:
            image = self.transform(image)
        else:
            # Normalize to [0, 1] following vincent1bt
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'anchor_targets': torch.from_numpy(anchor_targets).float(),
            'small_anchors': torch.from_numpy(small_anchors).float(),
            'big_anchors': torch.from_numpy(big_anchors).float()
        }


class LandmarkerDataset(BaseEarDataset):
    """Dataset for landmark training (images + keypoints)."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128),
        num_keypoints: int = 55
    ):
        """
        Args:
            npy_path: Path to NPY file
            root_dir: Root directory for images
            transform: Optional image transform
            target_size: Target image size
            num_keypoints: Number of keypoints to predict
        """
        self.num_keypoints = num_keypoints
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse landmark annotations."""
        self.keypoints = metadata.get('keypoints', [])
        self.visibility = metadata.get('visibility', None)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor
                - keypoints: [K, 2] tensor
                - visibility: [K] tensor (optional)
        """
        # Load image
        image = self._load_image(idx)
        
        # Get annotations
        keypoints = np.array(self.keypoints[idx]).reshape(-1, 2)
        annotations = {'keypoints': keypoints}
        
        # Resize image and annotations
        image, annotations = self._resize_image(image, annotations)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Normalize keypoints to [0, 1]
        kpts = annotations['keypoints'].reshape(-1, 2)
        kpts[:, 0] /= self.target_size[1]  # x / width
        kpts[:, 1] /= self.target_size[0]  # y / height
        
        sample = {
            'image': image,
            'keypoints': torch.from_numpy(kpts).float()
        }
        
        if self.visibility is not None:
            sample['visibility'] = torch.from_numpy(
                np.array(self.visibility[idx])
            ).float()
        
        return sample


class TeacherDataset(BaseEarDataset):
    """Dataset for teacher model training (images + boxes + landmarks)."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (128, 128)
    ):
        super().__init__(npy_path, root_dir, transform, target_size)
    
    def _parse_annotations(self, metadata: Dict):
        """Parse teacher annotations (both boxes and landmarks)."""
        self.bboxes = metadata.get('bboxes', [])
        self.keypoints = metadata.get('keypoints', [])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dict with:
                - image: [3, H, W] tensor
                - bboxes: [N, 4] tensor
                - keypoints: [N, K, 2] tensor
        """
        # Load image
        image = self._load_image(idx)
        
        # Get annotations
        annotations = {}
        if self.bboxes:
            annotations['bboxes'] = np.array(self.bboxes[idx])
        if self.keypoints:
            annotations['keypoints'] = np.array(self.keypoints[idx])
        
        # Resize image and annotations
        image, annotations = self._resize_image(image, annotations)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        sample = {'image': image}
        
        if 'bboxes' in annotations:
            sample['bboxes'] = torch.from_numpy(annotations['bboxes']).float()
        if 'keypoints' in annotations:
            sample['keypoints'] = torch.from_numpy(annotations['keypoints']).float()
        
        return sample


def collate_detector_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DetectorDataset.
    
    Stacks anchor-encoded targets which have fixed size (896, 5).
    
    Args:
        batch: List of sample dicts from DetectorDataset
        
    Returns:
        Dict with:
            - image: (B, 3, H, W)
            - anchor_targets: (B, 896, 5)
            - small_anchors: (B, 16, 16, 5)
            - big_anchors: (B, 8, 8, 5)
    """
    return {
        'image': torch.stack([sample['image'] for sample in batch]),
        'anchor_targets': torch.stack([sample['anchor_targets'] for sample in batch]),
        'small_anchors': torch.stack([sample['small_anchors'] for sample in batch]),
        'big_anchors': torch.stack([sample['big_anchors'] for sample in batch])
    }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-sized annotations.
    
    Args:
        batch: List of sample dicts
        
    Returns:
        Batched dict with padded annotations
    """
    # Stack images
    images = torch.stack([sample['image'] for sample in batch])
    
    result = {'image': images}
    
    # Handle bboxes (variable length per image)
    if 'bboxes' in batch[0]:
        # Find max number of boxes
        max_boxes = max(len(sample['bboxes']) for sample in batch)
        if max_boxes > 0:
            # Pad bboxes
            padded_bboxes = []
            for sample in batch:
                bboxes = sample['bboxes']
                if len(bboxes) < max_boxes:
                    padding = torch.zeros(max_boxes - len(bboxes), bboxes.shape[-1])
                    bboxes = torch.cat([bboxes, padding], dim=0)
                padded_bboxes.append(bboxes)
            result['bboxes'] = torch.stack(padded_bboxes)
            
            # Create mask for valid boxes
            result['bbox_mask'] = torch.stack([
                torch.cat([
                    torch.ones(len(sample['bboxes'])),
                    torch.zeros(max_boxes - len(sample['bboxes']))
                ])
                for sample in batch
            ])
    
    # Handle keypoints
    if 'keypoints' in batch[0]:
        result['keypoints'] = torch.stack([sample['keypoints'] for sample in batch])
    
    if 'visibility' in batch[0]:
        result['visibility'] = torch.stack([sample['visibility'] for sample in batch])
    
    return result


def get_dataloader(
    dataset_type: str,
    npy_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create a DataLoader.
    
    Args:
        dataset_type: One of 'detector', 'landmarker', 'teacher'
        npy_path: Path to NPY file
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for Dataset class
        
    Returns:
        DataLoader instance
    """
    dataset_map = {
        'detector': DetectorDataset,
        'landmarker': LandmarkerDataset,
        'teacher': TeacherDataset
    }
    
    collate_map = {
        'detector': collate_detector_fn,
        'landmarker': collate_fn,
        'teacher': collate_fn
    }
    
    if dataset_type not in dataset_map:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_map.keys())}")
    
    dataset = dataset_map[dataset_type](npy_path, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_map[dataset_type],
        pin_memory=True
    )
