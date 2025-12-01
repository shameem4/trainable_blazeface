"""
BlazeEar Detector - Inference script for ear detection.

Loads a trained checkpoint and runs inference on images.

Usage:
    from ear_detector.ear_detector import EarDetector
    
    detector = EarDetector("path/to/checkpoint.ckpt")
    detections = detector.detect("path/to/image.jpg")
    
    # Or with a numpy/tensor image
    detections = detector.detect(image_array)
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

# Add parent directory to path for standalone execution
script_dir = Path(__file__).parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

from ear_detector.model import BlazeEar
from ear_detector.lightning_module import BlazeEarLightningModule


class EarDetector:
    """
    Ear detector using trained BlazeEar model.
    
    Loads a checkpoint and provides inference capabilities for
    detecting ears in images.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        """
        Initialize the ear detector.
        
        Args:
            checkpoint_path: Path to trained .ckpt checkpoint file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            score_threshold: Minimum confidence score for detections
            nms_threshold: IoU threshold for NMS
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Get input size from model
        self.input_size = self.model.input_size
        
        # Normalization parameters (matching training)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def _load_model(self) -> BlazeEar:
        """Load model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load lightning module
        lightning_module = BlazeEarLightningModule.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=self.device,
        )
        
        # Extract the model
        return lightning_module.model
    
    def preprocess(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Image as path string, numpy array (H, W, C), or tensor
            
        Returns:
            Tuple of (preprocessed tensor, original size (height, width))
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy to tensor
        if isinstance(image, np.ndarray):
            original_size = (image.shape[0], image.shape[1])
            # Resize
            image = cv2.resize(image, (self.input_size, self.input_size))
            # Convert to tensor (H, W, C) -> (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                original_size = (image.shape[1], image.shape[2])
            else:
                original_size = (image.shape[2], image.shape[3])
            # Resize
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(image, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
            image = image.squeeze(0)
            if image.max() > 1.0:
                image = image / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Add batch dimension
        image = image.unsqueeze(0).to(self.device)
        
        # Normalize
        image = (image - self.mean) / self.std
        
        return image, original_size
    
    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Path, np.ndarray, torch.Tensor],
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run ear detection on an image.
        
        Args:
            image: Image as path, numpy array (H, W, C), or tensor
            return_raw: If True, also return raw model outputs
            
        Returns:
            Dictionary containing:
                - 'boxes': (N, 4) detected bounding boxes in [x1, y1, x2, y2] format
                           Coordinates are in original image scale
                - 'scores': (N,) confidence scores
                - 'num_detections': Number of detections
                - 'raw_outputs': (optional) Raw model outputs if return_raw=True
        """
        # Preprocess
        input_tensor, original_size = self.preprocess(image)
        orig_h, orig_w = original_size
        
        # Forward pass
        outputs = self.model(input_tensor)
        
        # Get scores and boxes
        scores = torch.sigmoid(outputs['classification'])  # (1, num_anchors, 1)
        boxes = self.model.decode_boxes(outputs['box_regression'])  # (1, num_anchors, 4)
        
        # Remove batch dimension
        scores = scores[0, :, 0]  # (num_anchors,)
        boxes = boxes[0]  # (num_anchors, 4)
        
        # Filter by score threshold
        mask = scores > self.score_threshold
        filtered_scores = scores[mask]
        filtered_boxes = boxes[mask]
        
        # Apply NMS
        if len(filtered_scores) > 0:
            keep = nms(filtered_boxes, filtered_scores, self.nms_threshold)
            final_boxes = filtered_boxes[keep]
            final_scores = filtered_scores[keep]
        else:
            final_boxes = torch.zeros((0, 4), device=self.device)
            final_scores = torch.zeros((0,), device=self.device)
        
        # Scale boxes to original image size
        # Model outputs are in normalized [0, 1] coordinates
        final_boxes[:, [0, 2]] *= orig_w  # x1, x2
        final_boxes[:, [1, 3]] *= orig_h  # y1, y2
        
        # Clamp to image bounds
        final_boxes[:, [0, 2]] = final_boxes[:, [0, 2]].clamp(0, orig_w)
        final_boxes[:, [1, 3]] = final_boxes[:, [1, 3]].clamp(0, orig_h)
        
        result = {
            'boxes': final_boxes,
            'scores': final_scores,
            'num_detections': len(final_scores),
        }
        
        if return_raw:
            result['raw_outputs'] = outputs
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run detection on a batch of images.
        
        Args:
            images: List of images (paths, arrays, or tensors)
            
        Returns:
            List of detection dictionaries, one per image
        """
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        detections: Optional[Dict[str, torch.Tensor]] = None,
        output_path: Optional[str] = None,
        show: bool = False,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image: Input image
            detections: Detection results (runs detect() if not provided)
            output_path: Optional path to save visualization
            show: Whether to display the image
            color: BGR color for bounding boxes
            thickness: Line thickness for boxes
            
        Returns:
            Image with drawn detections
        """
        # Load image if needed
        if isinstance(image, (str, Path)):
            vis_image = cv2.imread(str(image))
        else:
            vis_image = image.copy()
            if vis_image.shape[2] == 3 and vis_image.dtype == np.uint8:
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Run detection if not provided
        if detections is None:
            detections = self.detect(image)
        
        # Draw boxes
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw score label
            label = f'{score:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save if requested
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        # Show if requested
        if show:
            cv2.imshow('Ear Detection', vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return vis_image


def main():
    """Command-line interface for ear detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ear detection on images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to save output image')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Detection score threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--show', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    # Create detector
    detector = EarDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    
    # Run detection
    print(f"Running detection on: {args.image}")
    detections = detector.detect(args.image)
    
    print(f"Found {detections['num_detections']} ear(s)")
    for i, (box, score) in enumerate(zip(detections['boxes'], detections['scores'])):
        box = box.cpu().numpy()
        print(f"  Detection {i+1}: box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], score={score:.3f}")
    
    # Visualize
    if args.output or args.show:
        detector.visualize(
            args.image,
            detections,
            output_path=args.output,
            show=args.show,
        )
        if args.output:
            print(f"Saved visualization to: {args.output}")


if __name__ == '__main__':
    main()
