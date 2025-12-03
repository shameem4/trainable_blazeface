"""
Webcam demo for BlazeFace detection.
Detection only - no landmarks.

Supports loading:
- MediaPipe weights: blazeface.pth (raw state_dict)
- Retrained checkpoints: *.ckpt (dict with 'model_state_dict' key)
"""
import argparse
import cv2
import torch
import numpy as np
import sys
import time
from pathlib import Path
from threading import Thread


from blazeface import BlazeFace


class WebcamVideoStream:
    """Threaded webcam capture for better FPS.
    
    Based on: https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    """
    def __init__(self, src: int = 0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
    
    def start(self) -> "WebcamVideoStream":
        Thread(target=self._update, daemon=True).start()
        return self
    
    def _update(self) -> None:
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()
    
    def read(self) -> np.ndarray:
        return self.frame
    
    def stop(self) -> None:
        self.stopped = True
        self.stream.release()


class FPSCounter:
    """Simple FPS counter with rolling average."""
    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.times: list[float] = []
        self.fps = 0.0
    
    def tick(self) -> float:
        self.times.append(time.time())
        if len(self.times) > self.avg_frames:
            self.times.pop(0)
        if len(self.times) >= 2:
            self.fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])
        return self.fps
    
    def get(self) -> float:
        return self.fps


def draw_detections(
    img: np.ndarray,
    detections: torch.Tensor | np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Draw bounding boxes and confidence scores from detections.
    
    Args:
        img: Image to draw on (modified in place)
        detections: Detection tensor [N, 17] with format [ymin, xmin, ymax, xmax, ..., score]
        color: BGR color tuple
        thickness: Line thickness
    """
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if len(detections) == 0:
        return
        
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0])
        xmin = int(detections[i, 1])
        ymax = int(detections[i, 2])
        xmax = int(detections[i, 3])
        
        # Get confidence score (index 16)
        score = detections[i, 16] if detections.shape[1] > 16 else 0.0
        
        # Draw axis-aligned bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # Draw confidence score above the box
        label = f"{score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Position label above box, or inside if at top edge
        label_y = ymin - 5 if ymin > label_h + 5 else ymin + label_h + 5
        cv2.putText(
            img, label, (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def draw_fps(img: np.ndarray, fps: float) -> None:
    """Draw FPS counter on image."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 0), 2, cv2.LINE_AA)


def load_model(weights_path: str, device: torch.device) -> BlazeFace:
    """Load BlazeFace model from either MediaPipe weights or training checkpoint.
    
    Args:
        weights_path: Path to .pth (MediaPipe) or .ckpt (retrained) file
        device: Device to load model on
        
    Returns:
        Loaded BlazeFace model in eval mode
    """
    from blazebase import anchor_options, load_mediapipe_weights
    
    model = BlazeFace().to(device)
    
    # Check if this is a training checkpoint or MediaPipe weights
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Training checkpoint format - already in BlazeBlock format
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', None)
        print(f"Loaded training checkpoint (epoch {epoch})", end="")
        if val_loss is not None:
            print(f" - val_loss: {val_loss:.4f}")
        else:
            print()
    else:
        # MediaPipe weights format (BlazeBlock_WT) - needs conversion
        # Use load_mediapipe_weights which converts BlazeBlock_WT -> BlazeBlock
        missing, unexpected = load_mediapipe_weights(model, weights_path, strict=False)
        if missing:
            print(f"Warning: Missing keys: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys: {unexpected}")
        print("Loaded MediaPipe weights (converted from BlazeBlock_WT)")
    
    # Common setup for both formats
    model.eval()
    if hasattr(model, "generate_anchors"):
        model.generate_anchors(anchor_options)
    
    return model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Webcam demo for BlazeFace ear detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="model_weights/blazeface.pth",
        help="Path to weights file (.pth for MediaPipe, .ckpt for retrained). "
             "If not specified, uses model_weights/blazeface.pth"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirror mode (default: mirror enabled)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.9,
        help="Detection threshold (overrides model default)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Get the directory where this script is located for relative model paths
    SCRIPT_DIR = Path(__file__).parent

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {gpu}")
    torch.set_grad_enabled(False)

    # Determine weights path
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = str(SCRIPT_DIR / "model_weights" / "blazeface.pth")
    
    print(f"Loading weights: {weights_path}")
    
    # Load detector
    detector = load_model(weights_path, gpu)
    
    # Override detection threshold if specified
    if args.threshold is not None:
        detector.min_score_thresh = args.threshold
        print(f"Detection threshold: {args.threshold}")
    
    print("Model loaded")

    WINDOW = "BlazeFace Detection"
    cv2.namedWindow(WINDOW)
    
    # Use threaded webcam capture
    vs = WebcamVideoStream(src=args.camera).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    mirror_img = not args.no_mirror
    fps_counter = FPSCounter(avg_frames=30)
    
    print("Press 'q' or ESC to quit")
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
            
        # Convert BGR to RGB and optionally mirror
        if mirror_img:
            frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
        else:
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        # Run detection
        detections = detector.process(frame)

        # Draw detections
        draw_detections(frame, detections, color=(0, 255, 0), thickness=2)
        
        # Update and draw FPS
        fps = fps_counter.tick()
        draw_fps(frame, fps)

        # Display (convert RGB back to BGR for OpenCV)
        cv2.imshow(WINDOW, frame[:, :, ::-1])
        
        # Check for quit
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
            
    vs.stop()
    cv2.destroyAllWindows()
    sys.exit(0)