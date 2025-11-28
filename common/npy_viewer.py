"""
NPY Dataset Viewer

Visualize training data from NPY files with annotations.
Use arrow keys to navigate through images.
"""

import argparse
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Dict, Optional


class NPYViewer:
    """Interactive viewer for NPY dataset files."""

    def __init__(self, npy_path: str, root_dir: Optional[str] = None):
        """
        Initialize viewer.

        Args:
            npy_path: Path to NPY file
            root_dir: Optional root directory for image paths
        """
        self.npy_path = Path(npy_path)
        self.root_dir = Path(root_dir) if root_dir else None
        self.current_index = 0

        # Load metadata
        print(f"Loading {self.npy_path}...")
        metadata = np.load(self.npy_path, allow_pickle=True).item()

        self.image_paths = metadata['image_paths']
        self.data = metadata

        # Determine data type
        if 'bboxes' in metadata and isinstance(metadata['bboxes'][0], list):
            # Detector format: list of bboxes per image
            self.data_type = 'detector'
            self.bboxes = metadata['bboxes']
        elif 'bboxes' in metadata:
            # Teacher format: single bbox per image
            self.data_type = 'teacher'
            self.bboxes = metadata['bboxes']
        elif 'keypoints' in metadata:
            # Landmarker format
            self.data_type = 'landmarker'
            self.keypoints = metadata.get('keypoints', [])
        else:
            self.data_type = 'unknown'

        print(f"Loaded {len(self.image_paths)} samples")
        print(f"Data type: {self.data_type}")
        print(f"\nControls:")
        print("  Right Arrow / D: Next image")
        print("  Left Arrow / A: Previous image")
        print("  Q / ESC: Quit")
        print("  Space: Jump to image by index")

    def get_image_path(self, idx: int) -> Path:
        """Get absolute image path."""
        image_path = Path(self.image_paths[idx])

        if self.root_dir and not image_path.is_absolute():
            image_path = self.root_dir / image_path

        return image_path

    def draw_bbox(self, image: np.ndarray, bbox, color=(0, 255, 0), thickness=2):
        """Draw a bounding box on image."""
        if bbox is None:
            return

        # Handle different bbox formats
        try:
            # Convert to list/array if needed
            if isinstance(bbox, (list, tuple, np.ndarray)):
                bbox = np.array(bbox).flatten()
                if len(bbox) != 4:
                    return  # Invalid bbox
                x, y, w, h = bbox
            else:
                return  # Invalid bbox type

            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        except (ValueError, TypeError, IndexError):
            # Skip invalid bboxes
            pass

    def draw_keypoints(self, image: np.ndarray, keypoints: np.ndarray,
                       color=(0, 255, 255), radius=3):
        """Draw keypoints on image."""
        if keypoints is None or len(keypoints) == 0:
            return

        for kpt in keypoints:
            x, y = int(kpt[0]), int(kpt[1])
            # Check visibility if available
            visible = kpt[2] > 0 if len(kpt) > 2 else True

            if visible:
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1)

    def draw_annotations(self, image: np.ndarray, idx: int) -> np.ndarray:
        """Draw annotations on image based on data type."""
        if self.data_type == 'detector':
            # Draw all bboxes for this image
            bboxes = self.bboxes[idx]
            if bboxes:
                for bbox in bboxes:
                    self.draw_bbox(image, bbox, color=(0, 255, 0))

        elif self.data_type == 'teacher':
            # Draw single bbox
            bbox = self.bboxes[idx]
            self.draw_bbox(image, bbox, color=(255, 0, 0))

        elif self.data_type == 'landmarker':
            # Draw keypoints
            if idx < len(self.keypoints):
                kpts = self.keypoints[idx]
                if kpts is not None:
                    self.draw_keypoints(image, kpts)

                    # Also draw bbox if available
                    if 'bboxes' in self.data and idx < len(self.data['bboxes']):
                        bbox = self.data['bboxes'][idx]
                        if bbox:
                            self.draw_bbox(image, bbox[0] if isinstance(bbox, list) else bbox,
                                         color=(0, 255, 0), thickness=1)

        return image

    def get_annotation_info(self, idx: int) -> str:
        """Get annotation information as text."""
        info_lines = []

        try:
            if self.data_type == 'detector':
                bboxes = self.bboxes[idx]
                info_lines.append(f"Bboxes: {len(bboxes) if bboxes else 0}")
                if bboxes:
                    for i, bbox in enumerate(bboxes[:3]):  # Show first 3
                        try:
                            bbox = np.array(bbox).flatten()
                            if len(bbox) == 4:
                                info_lines.append(f"  [{i}] x={bbox[0]:.1f} y={bbox[1]:.1f} w={bbox[2]:.1f} h={bbox[3]:.1f}")
                        except (ValueError, TypeError, IndexError):
                            info_lines.append(f"  [{i}] Invalid bbox")
                    if len(bboxes) > 3:
                        info_lines.append(f"  ... and {len(bboxes) - 3} more")

            elif self.data_type == 'teacher':
                bbox = self.bboxes[idx]
                if bbox is not None:
                    try:
                        bbox = np.array(bbox).flatten()
                        if len(bbox) == 4:
                            info_lines.append(f"Bbox: x={bbox[0]:.1f} y={bbox[1]:.1f} w={bbox[2]:.1f} h={bbox[3]:.1f}")
                        else:
                            info_lines.append("Bbox: Invalid format")
                    except (ValueError, TypeError, IndexError):
                        info_lines.append("Bbox: Invalid")
                else:
                    info_lines.append("Bbox: None")

            elif self.data_type == 'landmarker':
                if idx < len(self.keypoints):
                    kpts = self.keypoints[idx]
                    if kpts is not None:
                        info_lines.append(f"Keypoints: {len(kpts)}")
                    else:
                        info_lines.append("Keypoints: None")
        except Exception as e:
            info_lines.append(f"Error: {str(e)}")

        return '\n'.join(info_lines) if info_lines else "No annotations"

    def show_image(self, idx: int):
        """Display image with annotations."""
        # Load image
        image_path = self.get_image_path(idx)

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load: {image_path}")
            return None

        # Draw annotations
        image = self.draw_annotations(image.copy(), idx)

        # Add info overlay
        info_text = [
            f"Image {idx + 1}/{len(self.image_paths)}",
            f"File: {image_path.name}",
            f"Size: {image.shape[1]}x{image.shape[0]}",
            f"Type: {self.data_type}",
            "",
            self.get_annotation_info(idx)
        ]

        # Draw semi-transparent background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150 + len(info_text) * 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Draw text
        y_offset = 30
        for line in info_text:
            cv2.putText(image, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return image

    def run(self):
        """Run the interactive viewer."""
        window_name = f"NPY Viewer - {self.npy_path.name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            # Show current image
            image = self.show_image(self.current_index)

            if image is not None:
                cv2.imshow(window_name, image)

            # Handle key press
            key = cv2.waitKey(0) & 0xFF

            # Right arrow or 'd' - next image
            if key == 83 or key == ord('d'):  # Right arrow
                self.current_index = (self.current_index + 1) % len(self.image_paths)

            # Left arrow or 'a' - previous image
            elif key == 81 or key == ord('a'):  # Left arrow
                self.current_index = (self.current_index - 1) % len(self.image_paths)

            # Space - jump to index
            elif key == ord(' '):
                cv2.destroyWindow(window_name)
                try:
                    idx = int(input(f"Enter index (0-{len(self.image_paths) - 1}): "))
                    if 0 <= idx < len(self.image_paths):
                        self.current_index = idx
                    else:
                        print(f"Invalid index. Must be 0-{len(self.image_paths) - 1}")
                except ValueError:
                    print("Invalid input")
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Q or ESC - quit
            elif key == ord('q') or key == 27:  # ESC
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='View NPY dataset files interactively')

    parser.add_argument('npy_file', type=str, nargs='?', default=None,
                       help='Path to NPY file (e.g., data/preprocessed/train_teacher.npy)')
    parser.add_argument('--root-dir', type=str, default=None,
                       help='Root directory for image paths (if using relative paths)')

    args = parser.parse_args()

    # If no file provided, open file chooser dialog
    npy_file = args.npy_file
    if not npy_file:
        root = tk.Tk()
        root.withdraw()
        npy_file = filedialog.askopenfilename(
            title='Select NPY dataset file',
            filetypes=[
                ('NPY Files', '*.npy'),
                ('All Files', '*.*')
            ],
            initialdir='data/preprocessed'
        )
        if not npy_file:
            print('No file selected.')
            return

    # Create and run viewer
    viewer = NPYViewer(npy_file, args.root_dir)
    viewer.run()


if __name__ == '__main__':
    main()
