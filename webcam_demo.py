"""
Webcam demo for BlazeFace detection.
Detection only - no landmarks.
"""
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
    """Draw bounding boxes from detections.
    
    Args:
        img: Image to draw on (modified in place)
        detections: Detection tensor [N, 4+] with format [ymin, xmin, ymax, xmax, ...]
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
        
        # Draw axis-aligned bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_fps(img: np.ndarray, fps: float) -> None:
    """Draw FPS counter on image."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == "__main__":
    # Get the directory where this script is located for relative model paths
    SCRIPT_DIR = Path(__file__).parent

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {gpu}")
    torch.set_grad_enabled(False)

    # Load detector
    detector = BlazeFace().to(gpu)
    detector.load_weights(str(SCRIPT_DIR / "model_weights" / "blazeface.pth"))
    detector.eval()
    print("Model loaded")

    WINDOW = "BlazeFace Detection"
    cv2.namedWindow(WINDOW)
    
    # Use threaded webcam capture
    vs = WebcamVideoStream(src=0).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    mirror_img = True
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