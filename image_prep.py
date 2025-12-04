from email.mime import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import csv
import cv2
import mediapipe as mp
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

try:
    import msvcrt  # Windows-only, used for non-blocking ESC detection
except ImportError:
    msvcrt = None

from tqdm import tqdm


import mediapipe as mp

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



def esc_pressed() -> bool:
    """Check if ESC key was pressed without blocking the processing loop."""
    if msvcrt:
        while msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b"\x1b":
                return True
            # Consume the second byte for special keys (e.g., arrows)
            if key in {b"\x00", b"\xe0"} and msvcrt.kbhit():
                msvcrt.getch()
        return False

    key = cv2.waitKey(1) & 0xFF
    return key == 27




if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_and_sort_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")
    print("Press ESC at any time to stop processing early.")

    current_idx = 0
    threshold = 0.95

    #  create file next to our csv_path - if it was train.csv create train_new.csv
    new_csv_path = csv_path.with_name(csv_path.stem + "_new" + csv_path.suffix)
    csv_file = new_csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_path", "x1", "y1", "w", "h"])
    print(f"Writing filtered detections to {new_csv_path}")

    processed_images: set[str] = set()
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)


    try:
        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as progress:
            while len(processed_images) < len(image_paths):
                image_path = image_paths[current_idx]

                if image_path not in processed_images:
                    processed_images.add(image_path)
                    progress.update(1)

                full_image_path = SCRIPT_DIR / args.data_root / image_path

                if not full_image_path.exists():
                    print(f"Warning: Image not found: {full_image_path}")
                    current_idx = (current_idx + 1) % len(image_paths)
                    continue

                img = cv2.imread(str(full_image_path))
                if img is None:
                    print(f"Warning: Failed to load image: {full_image_path}")
                    current_idx = (current_idx + 1) % len(image_paths)
                    continue

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
                results = face_detection.process(rgb_image)
                ih, iw, _ = img.shape
                if results.detections:
                    for detection in results.detections:
                        bbox_c = detection.location_data.relative_bounding_box
                        ih, iw, _ = img.shape
                        x1, y1, width, height = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), int(bbox_c.width * iw), int(bbox_c.height * ih)
                        score = detection.score[0]                        
                        if score < threshold:
                            continue
                        csv_writer.writerow([image_path, x1, y1, width, height])
 

                if esc_pressed():
                    print("ESC detected, stopping image processing loop.")
                    break

                current_idx = (current_idx + 1) % len(image_paths)
    finally:
        csv_file.close()

    print("done")
    cv2.destroyAllWindows()
    sys.exit(0)
