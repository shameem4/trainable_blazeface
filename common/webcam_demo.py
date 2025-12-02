"""
Webcam Demo - Real-time ear detection using webcam.

Loads a trained BlazeEar checkpoint and runs detection on webcam frames.
Uses matplotlib for display (works when OpenCV GUI is unavailable).

Usage:
    python common/webcam_demo.py --checkpoint models/ear_detector/checkpoint.ckpt
    
Controls:
    Close window or press 'q' - Quit
    Press 's' - Save current frame
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Add parent directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

from ear_detector.blazeear_inference import EarDetector


class WebcamDemo:
    """Real-time ear detection demo using webcam."""
    
    def __init__(
        self,
        checkpoint_path: str,
        camera_id: int = 0,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        device: str = None,
        window_name: str = "Ear Detection Demo",
    ):
        """
        Initialize the webcam demo.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            camera_id: Camera device ID (default 0)
            score_threshold: Detection confidence threshold
            nms_threshold: NMS IoU threshold
            device: Device for inference ('cuda', 'cpu', or None for auto)
            window_name: Name of the display window
        """
        self.camera_id = camera_id
        self.window_name = window_name
        self.last_frame = None
        self.running = False
        
        # FPS tracking
        self.fps_history = []
        self.fps_window = 30  # Average over last 30 frames
        
        # Load detector
        print(f"Loading checkpoint: {checkpoint_path}")
        self.detector = EarDetector(
            checkpoint_path=checkpoint_path,
            device=device,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        print(f"Model loaded on device: {self.detector.device}")
        print(f"Input size: {self.detector.input_size}x{self.detector.input_size}")
        
        # Matplotlib figure (will be created in run())
        self.fig = None
        self.ax = None
        self.im = None
        self.box_patches = []
        self.text_annotations = []
    
    def _update_frame(self, frame_num):
        """Update function for matplotlib animation."""
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            self.running = False
            return
        
        self.last_frame = frame.copy()
        start_time = time.perf_counter()
        
        # Convert BGR to RGB for detection and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        detections = self.detector.detect(frame_rgb)
        
        # Calculate frame time
        frame_time = time.perf_counter() - start_time
        fps = self._calculate_fps(frame_time)
        
        # Update image
        self.im.set_data(frame_rgb)
        
        # Clear previous boxes and annotations
        for patch in self.box_patches:
            patch.remove()
        self.box_patches.clear()
        
        for ann in self.text_annotations:
            ann.remove()
        self.text_annotations.clear()
        
        # Draw new detections
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            self.ax.add_patch(rect)
            self.box_patches.append(rect)
            
            # Draw label
            label = f'Ear {score:.2f}'
            text = self.ax.text(
                x1, y1 - 5, label,
                color='black', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.1', facecolor='lime', alpha=0.8)
            )
            self.text_annotations.append(text)
        
        # Update title with FPS and detection count
        self.ax.set_title(
            f'FPS: {fps:.1f} | Detections: {detections["num_detections"]} | Press Q to quit, S to save',
            fontsize=10
        )
        
        return [self.im] + self.box_patches + self.text_annotations
    
    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'q':
            print("\nQuitting...")
            self.running = False
            plt.close(self.fig)
        elif event.key == 's':
            if self.last_frame is not None:
                filename = self._save_frame(self.last_frame)
                print(f"Saved frame: {filename}")
    
    def _calculate_fps(self, frame_time: float) -> float:
        """Calculate smoothed FPS."""
        self.fps_history.append(frame_time)
        if len(self.fps_history) > self.fps_window:
            self.fps_history.pop(0)
        
        avg_time = sum(self.fps_history) / len(self.fps_history)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _save_frame(self, frame: np.ndarray) -> str:
        """Save current frame to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ear_detection_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
        return filename
    
    def run(self):
        """Run the webcam demo loop using matplotlib."""
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return
        
        # Get camera properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {width}x{height}")
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read initial frame")
            self.cap.release()
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = frame.copy()
        
        # Setup matplotlib figure
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.im = self.ax.imshow(frame_rgb)
        self.ax.axis('off')
        self.ax.set_title('Starting... Press Q to quit, S to save')
        
        # Connect keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self.running = True
        
        print("\nStarting webcam demo...")
        print("Press 'q' to quit, 's' to save frame")
        print("Close the window to exit")
        
        try:
            # Use FuncAnimation for smooth updates
            ani = FuncAnimation(
                self.fig, self._update_frame,
                interval=1,  # Update as fast as possible
                blit=False,
                cache_frame_data=False,
            )
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.running = False
            self.cap.release()
            plt.close('all')


def main():
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(
        description='Real-time ear detection using webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  q - Quit the demo
  s - Save current frame with detections
  Close window to exit
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='ear_detector/outputs/checkpoints/last.ckpt',
        help='Path to trained model checkpoint (.ckpt file)',
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)',
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--nms_threshold',
        type=float,
        default=0.3,
        help='NMS IoU threshold (default: 0.3)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device for inference (default: auto)',
    )
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create and run demo
    demo = WebcamDemo(
        checkpoint_path=args.checkpoint,
        camera_id=args.camera,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device,
    )
    
    demo.run()


if __name__ == '__main__':
    main()
