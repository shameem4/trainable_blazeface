"""
Anchor Data Viewer - Visualize anchor matching for ear detector training.

Displays images with:
1. Ground truth bounding boxes (green)
2. All anchors (light gray, optional - press 'G' to toggle)
3. Closest anchor per GT (red crosshair)
4. Other positive anchors within distance threshold (orange crosshair)

Uses CENTER DISTANCE matching (not IoU) for BlazeFace-style unit anchors.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ear_detector.model import BlazeEar
from ear_detector.losses import DetectionLoss, compute_iou


class AnchorDataViewer:
    """Interactive viewer for anchor matching visualization."""
    
    def __init__(
        self,
        npy_path: str,
        root_dir: Optional[str] = None,
        pos_iou_threshold: float = 0.1,   # BlazeFace-style low threshold
        neg_iou_threshold: float = 0.05,  # Lower than pos to create ignore zone
        image_size: int = 128,
    ):
        """
        Initialize viewer.
        
        Args:
            npy_path: Path to training data NPY file (train_detector.npy)
            root_dir: Root directory for image paths
            pos_iou_threshold: IoU threshold for positive anchors
            neg_iou_threshold: IoU threshold for negative anchors
            image_size: Image size (128 for BlazeEar)
        """
        self.npy_path = Path(npy_path)
        self.root_dir = Path(root_dir) if root_dir else Path(".")
        self.image_size = image_size
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.current_index = 0
        self.show_all_anchors = False
        
        # Load data
        print(f"Loading {self.npy_path}...")
        metadata = np.load(self.npy_path, allow_pickle=True).item()
        self.image_paths = metadata['image_paths']
        self.bboxes = metadata['bboxes']
        
        print(f"Loaded {len(self.image_paths)} samples")
        
        # Create model to get anchors (BlazeEar generates BlazeFace-style unit anchors internally)
        self.model = BlazeEar(
            num_anchors_16=2,
            num_anchors_8=6,
        )
        self.anchors = self.model.anchors  # (N, 4) in [cx, cy, w, h] format
        print(f"Total anchors: {len(self.anchors)} (BlazeFace-style unit anchors)")
        
        # Create loss function for matching
        self.loss_fn = DetectionLoss(
            pos_iou_threshold=pos_iou_threshold,
            neg_iou_threshold=neg_iou_threshold,
        )
        
        self._print_controls()
    
    def _print_controls(self):
        """Print keyboard controls."""
        print("\n=== Controls ===")
        print("  Right Arrow / D: Next image")
        print("  Left Arrow / A: Previous image")
        print("  Space: Jump to index")
        print("  G: Toggle all anchors grid")
        print("  Q / ESC: Quit")
        print()
    
    def get_image_path(self, idx: int) -> Path:
        """Get absolute image path."""
        image_path = Path(self.image_paths[idx])
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        return image_path
    
    def normalize_bboxes(self, bboxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Convert bboxes from x,y,w,h pixel format to normalized x1,y1,x2,y2.
        
        Args:
            bboxes: (N, 4) array in x,y,w,h pixel coordinates
            img_width: Original image width
            img_height: Original image height
            
        Returns:
            (N, 4) array in normalized x1,y1,x2,y2 format
        """
        bboxes = bboxes.reshape(-1, 4).astype(np.float32)
        normalized = np.zeros_like(bboxes)
        
        # x,y,w,h -> x1,y1,x2,y2 normalized
        normalized[:, 0] = bboxes[:, 0] / img_width  # x1
        normalized[:, 1] = bboxes[:, 1] / img_height  # y1
        normalized[:, 2] = (bboxes[:, 0] + bboxes[:, 2]) / img_width  # x2
        normalized[:, 3] = (bboxes[:, 1] + bboxes[:, 3]) / img_height  # y2
        
        return np.clip(normalized, 0, 1)
    
    def match_anchors_for_image(self, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Match anchors to GT boxes using CENTER DISTANCE (not IoU).
        
        For BlazeFace-style unit anchors, IoU-based matching doesn't work well
        because all unit anchors have similar IoU with small boxes.
        
        Args:
            gt_boxes: (M, 4) ground truth boxes in x1,y1,x2,y2 normalized
            
        Returns:
            matched_labels: (N,) 1=positive, 0=negative
            distances: (N,) distance to closest GT center
            best_anchor_per_gt: List of anchor indices that are closest to each GT
        """
        num_anchors = self.anchors.shape[0]
        
        if gt_boxes.shape[0] == 0:
            return (
                torch.zeros(num_anchors),
                torch.ones(num_anchors),  # Max distance
                [],
            )
        
        # Compute GT box centers
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2  # (M,)
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2  # (M,)
        
        # Compute distance from each anchor center to each GT center
        anchor_cx = self.anchors[:, 0].unsqueeze(1)  # (N, 1)
        anchor_cy = self.anchors[:, 1].unsqueeze(1)  # (N, 1)
        gt_cx_expanded = gt_cx.unsqueeze(0)  # (1, M)
        gt_cy_expanded = gt_cy.unsqueeze(0)  # (1, M)
        
        distances = torch.sqrt((anchor_cx - gt_cx_expanded)**2 + (anchor_cy - gt_cy_expanded)**2)  # (N, M)
        
        # Find closest GT for each anchor
        min_dist_per_anchor, best_gt_idx = distances.min(dim=1)  # (N,)
        
        # Initialize all as negative
        matched_labels = torch.zeros(num_anchors, dtype=torch.float32)
        
        # Find closest anchor for each GT (always positive)
        best_anchor_per_gt = distances.argmin(dim=0).tolist()  # (M,)
        
        # Mark closest anchors as positive
        for anchor_idx in best_anchor_per_gt:
            matched_labels[anchor_idx] = 1
        
        # Mark additional nearby anchors as positive (within threshold)
        pos_dist_threshold = self.pos_iou_threshold  # Reuse as distance threshold
        for gt_idx in range(gt_boxes.shape[0]):
            close_mask = distances[:, gt_idx] < pos_dist_threshold
            matched_labels[close_mask] = 1
        
        return matched_labels, min_dist_per_anchor, best_anchor_per_gt
    
    def draw_anchor_point(self, image: np.ndarray, anchor: torch.Tensor, 
                          color: Tuple[int, int, int], radius: int = 5,
                          label: Optional[str] = None):
        """
        Draw an anchor center point on the image.
        
        For BlazeFace-style unit anchors, we draw the center point with a crosshair
        since the anchor size (w=h=1) covers the whole image.
        """
        h, w = image.shape[:2]
        
        # Get anchor center
        cx, cy = anchor[0].item(), anchor[1].item()
        px = int(cx * w)
        py = int(cy * h)
        
        # Draw filled circle at center
        cv2.circle(image, (px, py), radius, color, -1)
        
        # Draw crosshair
        cross_size = radius + 5
        cv2.line(image, (px - cross_size, py), (px + cross_size, py), color, 2)
        cv2.line(image, (px, py - cross_size), (px, py + cross_size), color, 2)
        
        if label:
            cv2.putText(image, label, (px + 8, py - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_anchor_box(self, image: np.ndarray, anchor: torch.Tensor, 
                        color: Tuple[int, int, int], thickness: int = 1,
                        label: Optional[str] = None):
        """Draw an anchor box on the image (for sized anchors only)."""
        h, w = image.shape[:2]
        
        # Convert from [cx, cy, w, h] normalized to pixel coordinates
        cx, cy, aw, ah = anchor.tolist()
        x1 = int((cx - aw / 2) * w)
        y1 = int((cy - ah / 2) * h)
        x2 = int((cx + aw / 2) * w)
        y2 = int((cy + ah / 2) * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        if label:
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_gt_box(self, image: np.ndarray, bbox: np.ndarray,
                    color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2):
        """Draw ground truth box on image (normalized x1,y1,x2,y2)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    def show_image(self, idx: int) -> Optional[np.ndarray]:
        """Display image with anchor matching visualization."""
        # Load image
        image_path = self.get_image_path(idx)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return None
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize to match model input
        display_size = 512  # Display at larger size for visibility
        image = cv2.resize(image, (display_size, display_size))
        
        # Get bboxes and normalize
        bboxes_raw = self.bboxes[idx].reshape(-1, 4)
        gt_boxes_norm = self.normalize_bboxes(bboxes_raw, orig_w, orig_h)
        gt_boxes = torch.tensor(gt_boxes_norm, dtype=torch.float32)
        
        # Match anchors (using center distance, not IoU)
        matched_labels, distances, best_anchor_per_gt = self.match_anchors_for_image(gt_boxes)
        
        # Draw all anchors (optional, very faint) - show as small dots for unit anchors
        if self.show_all_anchors:
            for i, anchor in enumerate(self.anchors):
                h, w = image.shape[:2]
                cx, cy = anchor[0].item(), anchor[1].item()
                cv2.circle(image, (int(cx * w), int(cy * h)), 2, (50, 50, 50), -1)
        
        # Draw positive anchors (excluding best-per-GT which we draw separately)
        pos_mask = matched_labels == 1
        pos_indices = pos_mask.nonzero(as_tuple=True)[0].tolist()
        best_set = set(best_anchor_per_gt)
        
        for i in pos_indices:
            if i not in best_set:
                dist = distances[i].item()
                self.draw_anchor_point(image, self.anchors[i], (255, 100, 0), 6,
                                       f"d:{dist:.2f}")  # Orange
        
        # Draw best anchor per GT (red, larger)
        for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
            dist = distances[anchor_idx].item()
            self.draw_anchor_point(image, self.anchors[anchor_idx], (0, 0, 255), 8,
                                   f"Best#{gt_idx} d:{dist:.3f}")  # Red
        
        # Draw ground truth boxes (green, on top)
        for i, gt_box in enumerate(gt_boxes_norm):
            self.draw_gt_box(image, gt_box, (0, 255, 0), 3)
        
        # Calculate stats
        num_pos = int(pos_mask.sum().item())
        num_neg = int((matched_labels == 0).sum().item())
        avg_pos_dist = distances[pos_mask].mean().item() if num_pos > 0 else 0
        
        # Draw info overlay
        info_lines = [
            f"Image {idx + 1}/{len(self.image_paths)}",
            f"File: {image_path.name}",
            f"Original: {orig_w}x{orig_h}",
            f"",
            f"GT Boxes: {len(gt_boxes)}",
            f"",
            f"Positive anchors: {num_pos}",
            f"Negative anchors: {num_neg}",
            f"Avg positive distance: {avg_pos_dist:.3f}",
            f"",
            f"Distance threshold: {self.pos_iou_threshold}",
            f"",
            f"Legend:",
            f"  Green box = GT boxes",
            f"  Red + = Closest anchor center per GT",
            f"  Orange + = Other positive anchor centers",
            f"",
            f"Note: Using CENTER DISTANCE matching",
            f"(not IoU - unit anchors all have same IoU)",
            f"",
            f"Press 'G' to toggle anchor grid",
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (350, 30 + len(info_lines) * 18), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw text
        y_offset = 25
        for line in info_lines:
            cv2.putText(image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 18
        
        return image
    
    def run(self):
        """Run the interactive viewer."""
        window_name = "Anchor Data Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        
        while True:
            image = self.show_image(self.current_index)
            
            if image is not None:
                cv2.imshow(window_name, image)
            
            key = cv2.waitKey(0) & 0xFF
            
            # Right arrow or 'd'
            if key == 83 or key == ord('d'):
                self.current_index = (self.current_index + 1) % len(self.image_paths)
            
            # Left arrow or 'a'
            elif key == 81 or key == ord('a'):
                self.current_index = (self.current_index - 1) % len(self.image_paths)
            
            # 'g' - toggle anchor grid
            elif key == ord('g'):
                self.show_all_anchors = not self.show_all_anchors
                print(f"Show all anchors: {self.show_all_anchors}")
            
            # Space - jump to index
            elif key == ord(' '):
                cv2.destroyWindow(window_name)
                try:
                    idx = int(input(f"Enter index (0-{len(self.image_paths) - 1}): "))
                    if 0 <= idx < len(self.image_paths):
                        self.current_index = idx
                    else:
                        print(f"Invalid index")
                except ValueError:
                    print("Invalid input")
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 800, 800)
            
            # Q or ESC
            elif key == ord('q') or key == 27:
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize anchor matching for ear detector training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anchor_data_view.py
  python anchor_data_view.py --npy data/preprocessed/train_detector.npy
  python anchor_data_view.py --pos-iou 0.1 --neg-iou 0.05
        """
    )
    
    parser.add_argument('--npy', type=str, 
                       default='data/preprocessed/train_detector.npy',
                       help='Path to training data NPY file')
    parser.add_argument('--root-dir', type=str, default='.',
                       help='Root directory for image paths')
    parser.add_argument('--pos-iou', type=float, default=0.1,
                       help='Positive IoU threshold (default: 0.1 for BlazeFace-style unit anchors)')
    parser.add_argument('--neg-iou', type=float, default=0.05,
                       help='Negative IoU threshold (default: 0.05)')
    
    args = parser.parse_args()
    
    viewer = AnchorDataViewer(
        npy_path=args.npy,
        root_dir=args.root_dir,
        pos_iou_threshold=args.pos_iou,
        neg_iou_threshold=args.neg_iou,
    )
    viewer.run()


if __name__ == '__main__':
    main()
