import cv2
import mediapipe as mp
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

# Initialize MediaPipe Face Detector
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def load_and_sort_csv(csv_path: str) -> tuple[list[str], dict[str, list[tuple[int, int, int, int]]]]:
    """Load train.csv and sort by image_path.

    Returns:
        - List of unique image paths (sorted)
        - Dictionary mapping image_path to list of bounding boxes (x1, y1, w, h)
    """
    df = pd.read_csv(csv_path)

    # Sort by image_path
    df = df.sort_values('image_path')

    # Group boxes by image_path (handle multiple faces per image)
    image_to_boxes = defaultdict(list)

    for _, row in df.iterrows():
        image_path = row['image_path']
        x1, y1, w, h = int(row['x1']), int(row['y1']), int(row['w']), int(row['h'])
        image_to_boxes[image_path].append((x1, y1, w, h))

    # Get sorted unique image paths
    unique_image_paths = sorted(image_to_boxes.keys())

    return unique_image_paths, dict(image_to_boxes)




def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image demo for BlazeFace detection with CSV comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--csv", "-c",
        type=str,
        default="data/splits/train.csv",
        help="Path to CSV file with image annotations"
    )
    parser.add_argument(
        "--data-root", "-d",
        type=str,
        default="data/raw/blazeface",
        help="Root directory for image paths (prepended to CSV image_path)"
    )



    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent



    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_and_sort_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")

 
    current_idx = 0
    while True:
        # Get current image info
        image_path = image_paths[current_idx]
        gt_boxes = image_to_boxes[image_path]

        # Construct full path
        full_image_path = SCRIPT_DIR / args.data_root / image_path

        # Load image
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            continue

        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"Warning: Failed to load image: {full_image_path}")

            continue

        # Convert BGR to RGB for detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run detection
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_rgb)

        # Draw face detections on the original image
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(img, detection)

        cv2.imshow('BlazeFace Face Detection Demo', img)
        current_idx += 1
        if current_idx >= len(image_paths):
            break
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == ord('a') or key == 81:  # 'a' or Left arrow
            current_idx = (current_idx - 1) % len(image_paths)
            print(f"Previous: {current_idx + 1}/{len(image_paths)}")
        elif key == ord('d') or key == 83:  # 'd' or Right arrow
            current_idx = (current_idx + 1) % len(image_paths)
            print(f"Next: {current_idx + 1}/{len(image_paths)}")

    cv2.destroyAllWindows()
    sys.exit(0)
