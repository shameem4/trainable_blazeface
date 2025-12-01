"""
Bounding box validation and processing utilities.

This module provides reusable functions for validating and processing
bounding boxes across the codebase (preprocessor, datasets, etc.).

Uses torchvision.ops for core operations where possible to reduce custom code.

Supported bbox formats:
- xywh: [x, y, width, height] - top-left corner + dimensions
- xyxy: [x1, y1, x2, y2] - top-left and bottom-right corners
- cxcywh: [cx, cy, width, height] - center + dimensions
- normalized: coordinates in [0, 1] range (relative to image size)
- absolute: coordinates in pixel values
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torchvision.ops as tv_ops


# ============================================================================
# Torchvision wrapper functions (preferred for batch operations)
# ============================================================================

def box_convert(
    boxes: Union[torch.Tensor, np.ndarray, List],
    in_fmt: str,
    out_fmt: str,
) -> torch.Tensor:
    """
    Convert boxes between formats using torchvision.ops.box_convert.
    
    Args:
        boxes: Bounding boxes as tensor, numpy array, or list
        in_fmt: Input format ('xyxy', 'xywh', 'cxcywh')
        out_fmt: Output format ('xyxy', 'xywh', 'cxcywh')
        
    Returns:
        Converted boxes as torch.Tensor
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    return tv_ops.box_convert(boxes, in_fmt, out_fmt)


def box_area(boxes: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
    """
    Compute area of boxes using torchvision.ops.box_area.
    
    Args:
        boxes: Bounding boxes in xyxy format
        
    Returns:
        Areas as torch.Tensor
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    return tv_ops.box_area(boxes)


def clip_boxes_to_image(
    boxes: Union[torch.Tensor, np.ndarray, List],
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """
    Clip boxes to image boundaries using torchvision.ops.clip_boxes_to_image.
    
    Args:
        boxes: Bounding boxes in xyxy format
        image_height: Image height in pixels
        image_width: Image width in pixels
        
    Returns:
        Clipped boxes as torch.Tensor
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    return tv_ops.clip_boxes_to_image(boxes, size=(image_height, image_width))


def remove_small_boxes(
    boxes: Union[torch.Tensor, np.ndarray, List],
    min_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove boxes smaller than min_size using torchvision.ops.remove_small_boxes.
    
    Args:
        boxes: Bounding boxes in xyxy format
        min_size: Minimum size (boxes with width OR height < min_size are removed)
        
    Returns:
        Tuple of (filtered_boxes, keep_indices)
    """
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)
    keep_indices = tv_ops.remove_small_boxes(boxes, min_size=min_size)
    return boxes[keep_indices], keep_indices


def box_iou(
    boxes1: Union[torch.Tensor, np.ndarray, List],
    boxes2: Union[torch.Tensor, np.ndarray, List],
) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes using torchvision.ops.box_iou.
    
    Args:
        boxes1: First set of boxes in xyxy format [N, 4]
        boxes2: Second set of boxes in xyxy format [M, 4]
        
    Returns:
        IoU matrix [N, M]
    """
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.as_tensor(boxes1, dtype=torch.float32)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.as_tensor(boxes2, dtype=torch.float32)
    if boxes1.ndim == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.ndim == 1:
        boxes2 = boxes2.unsqueeze(0)
    return tv_ops.box_iou(boxes1, boxes2)


# ============================================================================
# BBoxChecker class for validation (uses torchvision internally)
# ============================================================================

class BBoxChecker:
    """
    Modular bounding box validator with configurable options.
    
    Uses torchvision.ops internally for efficient operations.
    Can be used in preprocessing pipelines and dataset loaders
    to ensure consistent bbox validation across the codebase.
    """
    
    def __init__(
        self,
        min_width: float = 0.0,
        min_height: float = 0.0,
        min_area: float = 0.0,
        allow_negative_coords: bool = False,
        clamp_to_bounds: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        """
        Initialize bbox checker with validation parameters.
        
        Args:
            min_width: Minimum allowed width (0 = any positive width)
            min_height: Minimum allowed height (0 = any positive height)
            min_area: Minimum allowed area (width * height)
            allow_negative_coords: Whether to allow negative x, y coordinates
            clamp_to_bounds: Whether to clamp coords to image bounds
            image_width: Image width for bounds checking (required if clamp_to_bounds)
            image_height: Image height for bounds checking (required if clamp_to_bounds)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.allow_negative_coords = allow_negative_coords
        self.clamp_to_bounds = clamp_to_bounds
        self.image_width = image_width
        self.image_height = image_height
        
        if clamp_to_bounds and (image_width is None or image_height is None):
            raise ValueError("image_width and image_height required when clamp_to_bounds=True")
    
    def is_valid_xywh(self, bbox: Union[List, Tuple, np.ndarray]) -> bool:
        """
        Validate bbox in x, y, width, height format.
        
        Args:
            bbox: [x, y, w, h] bounding box
            
        Returns:
            True if valid, False otherwise
        """
        if bbox is None:
            return False
        
        try:
            if len(bbox) != 4:
                return False
            
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # Check dimensions
            if w <= self.min_width or h <= self.min_height:
                return False
            
            # Check area
            if w * h < self.min_area:
                return False
            
            # Check coordinates
            if not self.allow_negative_coords and (x < 0 or y < 0):
                return False
            
            # Check bounds
            if self.clamp_to_bounds:
                if x >= self.image_width or y >= self.image_height:
                    return False
                if x + w <= 0 or y + h <= 0:
                    return False
            
            return True
            
        except (TypeError, ValueError, IndexError):
            return False
    
    def is_valid_xyxy(self, bbox: Union[List, Tuple, np.ndarray]) -> bool:
        """
        Validate bbox in x1, y1, x2, y2 format.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            True if valid, False otherwise
        """
        if bbox is None:
            return False
        
        try:
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # x2 must be greater than x1, y2 must be greater than y1
            w = x2 - x1
            h = y2 - y1
            
            if w <= self.min_width or h <= self.min_height:
                return False
            
            if w * h < self.min_area:
                return False
            
            if not self.allow_negative_coords and (x1 < 0 or y1 < 0):
                return False
            
            if self.clamp_to_bounds:
                if x1 >= self.image_width or y1 >= self.image_height:
                    return False
                if x2 <= 0 or y2 <= 0:
                    return False
            
            return True
            
        except (TypeError, ValueError, IndexError):
            return False
    
    def is_valid_normalized(self, bbox: Union[List, Tuple, np.ndarray]) -> bool:
        """
        Validate normalized bbox [x1, y1, x2, y2] where all values in [0, 1].
        
        Args:
            bbox: Normalized [x1, y1, x2, y2] bounding box
            
        Returns:
            True if valid, False otherwise
        """
        if bbox is None:
            return False
        
        try:
            if len(bbox) != 4:
                return False
            
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # All values must be in [0, 1]
            if not (0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1):
                return False
            
            # x2 > x1 and y2 > y1
            if x2 <= x1 or y2 <= y1:
                return False
            
            return True
            
        except (TypeError, ValueError, IndexError):
            return False
    
    def clamp_xywh(
        self, 
        bbox: Union[List, Tuple, np.ndarray],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Optional[List[float]]:
        """
        Clamp bbox to image bounds using torchvision, returning None if invalid.
        
        Args:
            bbox: [x, y, w, h] bounding box
            image_width: Image width (uses self.image_width if None)
            image_height: Image height (uses self.image_height if None)
            
        Returns:
            Clamped [x, y, w, h] or None if invalid
        """
        img_w = image_width or self.image_width
        img_h = image_height or self.image_height
        
        if img_w is None or img_h is None:
            raise ValueError("image_width and image_height required for clamping")
        
        try:
            # Convert to xyxy, clamp, convert back
            boxes_xywh = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32)
            boxes_xyxy = tv_ops.box_convert(boxes_xywh, 'xywh', 'xyxy')
            clipped = tv_ops.clip_boxes_to_image(boxes_xyxy, size=(img_h, img_w))
            result_xywh = tv_ops.box_convert(clipped, 'xyxy', 'xywh')
            
            x, y, w, h = result_xywh[0].tolist()
            
            # Check if valid after clamping
            if w <= self.min_width or h <= self.min_height:
                return None
            
            return [x, y, w, h]
            
        except (TypeError, ValueError, IndexError):
            return None
    
    def clamp_xyxy(
        self,
        bbox: Union[List, Tuple, np.ndarray],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> Optional[List[float]]:
        """
        Clamp bbox to image bounds using torchvision, returning None if invalid.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            image_width: Image width (uses self.image_width if None)
            image_height: Image height (uses self.image_height if None)
            
        Returns:
            Clamped [x1, y1, x2, y2] or None if invalid
        """
        img_w = image_width or self.image_width
        img_h = image_height or self.image_height
        
        if img_w is None or img_h is None:
            raise ValueError("image_width and image_height required for clamping")
        
        try:
            boxes = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32)
            clipped = tv_ops.clip_boxes_to_image(boxes, size=(img_h, img_w))
            
            x1, y1, x2, y2 = clipped[0].tolist()
            
            # Check if valid after clamping
            if x2 - x1 <= self.min_width or y2 - y1 <= self.min_height:
                return None
            
            return [x1, y1, x2, y2]
            
        except (TypeError, ValueError, IndexError):
            return None


# ============================================================================
# Convenience functions for common use cases
# ============================================================================

def is_valid_bbox_xywh(bbox, min_width: float = 0, min_height: float = 0) -> bool:
    """
    Quick validation for xywh format bbox.
    
    Args:
        bbox: [x, y, w, h] bounding box
        min_width: Minimum width threshold
        min_height: Minimum height threshold
        
    Returns:
        True if valid
    """
    checker = BBoxChecker(min_width=min_width, min_height=min_height)
    return checker.is_valid_xywh(bbox)


def is_valid_bbox_xyxy(bbox, min_width: float = 0, min_height: float = 0) -> bool:
    """
    Quick validation for xyxy format bbox.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        min_width: Minimum width threshold
        min_height: Minimum height threshold
        
    Returns:
        True if valid
    """
    checker = BBoxChecker(min_width=min_width, min_height=min_height)
    return checker.is_valid_xyxy(bbox)


def is_valid_bbox_normalized(bbox) -> bool:
    """
    Quick validation for normalized xyxy format bbox.
    
    Args:
        bbox: Normalized [x1, y1, x2, y2] bounding box (values in [0,1])
        
    Returns:
        True if valid
    """
    checker = BBoxChecker()
    return checker.is_valid_normalized(bbox)


def xywh_to_xyxy(bbox: Union[List, Tuple, np.ndarray]) -> List[float]:
    """
    Convert bbox from [x, y, w, h] to [x1, y1, x2, y2] format using torchvision.
    
    Args:
        bbox: [x, y, w, h] bounding box
        
    Returns:
        [x1, y1, x2, y2] bounding box
    """
    result = box_convert(bbox, 'xywh', 'xyxy')
    return result[0].tolist()


def xyxy_to_xywh(bbox: Union[List, Tuple, np.ndarray]) -> List[float]:
    """
    Convert bbox from [x1, y1, x2, y2] to [x, y, w, h] format using torchvision.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box
        
    Returns:
        [x, y, w, h] bounding box
    """
    result = box_convert(bbox, 'xyxy', 'xywh')
    return result[0].tolist()


def normalize_bbox_xywh(
    bbox: Union[List, Tuple, np.ndarray],
    image_width: int,
    image_height: int,
    clamp: bool = True,
) -> Optional[List[float]]:
    """
    Normalize xywh bbox to [0, 1] range and convert to xyxy format.
    Uses torchvision for conversion and clamping.
    
    Args:
        bbox: [x, y, w, h] bounding box in absolute coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        clamp: Whether to clamp values to [0, 1]
        
    Returns:
        Normalized [x1, y1, x2, y2] or None if invalid after normalization
    """
    try:
        # Convert to tensor and xyxy format
        boxes_xywh = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32)
        
        # Check for invalid dimensions before conversion
        if boxes_xywh[0, 2] <= 0 or boxes_xywh[0, 3] <= 0:
            return None
        
        boxes_xyxy = tv_ops.box_convert(boxes_xywh, 'xywh', 'xyxy')
        
        if clamp:
            # Clip to image bounds
            boxes_xyxy = tv_ops.clip_boxes_to_image(boxes_xyxy, size=(image_height, image_width))
        
        # Normalize
        x1, y1, x2, y2 = boxes_xyxy[0].tolist()
        x1_norm = x1 / image_width
        y1_norm = y1 / image_height
        x2_norm = x2 / image_width
        y2_norm = y2 / image_height
        
        # Ensure valid after clamping/normalization
        if x2_norm <= x1_norm or y2_norm <= y1_norm:
            return None
        
        return [x1_norm, y1_norm, x2_norm, y2_norm]
        
    except (TypeError, ValueError, IndexError):
        return None


def normalize_bbox_xyxy(
    bbox: Union[List, Tuple, np.ndarray],
    image_width: int,
    image_height: int,
    clamp: bool = True,
) -> Optional[List[float]]:
    """
    Normalize xyxy bbox to [0, 1] range.
    Uses torchvision for clamping.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box in absolute coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        clamp: Whether to clamp values to [0, 1]
        
    Returns:
        Normalized [x1, y1, x2, y2] or None if invalid after normalization
    """
    try:
        boxes = torch.tensor([[bbox[0], bbox[1], bbox[2], bbox[3]]], dtype=torch.float32)
        
        # Check for invalid boxes
        if boxes[0, 2] <= boxes[0, 0] or boxes[0, 3] <= boxes[0, 1]:
            return None
        
        if clamp:
            boxes = tv_ops.clip_boxes_to_image(boxes, size=(image_height, image_width))
        
        # Normalize
        x1, y1, x2, y2 = boxes[0].tolist()
        x1_norm = x1 / image_width
        y1_norm = y1 / image_height
        x2_norm = x2 / image_width
        y2_norm = y2 / image_height
        
        # Ensure valid after clamping/normalization
        if x2_norm <= x1_norm or y2_norm <= y1_norm:
            return None
        
        return [x1_norm, y1_norm, x2_norm, y2_norm]
        
    except (TypeError, ValueError, IndexError):
        return None


def filter_valid_bboxes(
    bboxes: List,
    format: str = 'xywh',
    min_width: float = 0,
    min_height: float = 0,
) -> List:
    """
    Filter a list of bboxes, keeping only valid ones.
    
    Args:
        bboxes: List of bounding boxes
        format: 'xywh' or 'xyxy'
        min_width: Minimum width threshold
        min_height: Minimum height threshold
        
    Returns:
        List of valid bboxes
    """
    checker = BBoxChecker(min_width=min_width, min_height=min_height)
    
    if format == 'xywh':
        return [b for b in bboxes if checker.is_valid_xywh(b)]
    elif format == 'xyxy':
        return [b for b in bboxes if checker.is_valid_xyxy(b)]
    else:
        raise ValueError(f"Unknown format: {format}. Use 'xywh' or 'xyxy'")


def filter_small_boxes_batch(
    boxes: Union[torch.Tensor, np.ndarray, List],
    min_size: float,
) -> torch.Tensor:
    """
    Efficiently filter small boxes from a batch using torchvision.
    
    Args:
        boxes: Bounding boxes in xyxy format [N, 4]
        min_size: Minimum size threshold
        
    Returns:
        Filtered boxes tensor
    """
    filtered, _ = remove_small_boxes(boxes, min_size)
    return filtered
