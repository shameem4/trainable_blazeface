import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

try:
    import msvcrt  # Windows-only, used for non-blocking ESC detection
except ImportError:
    msvcrt = None

from tqdm import tqdm

from retinaface import RetinaFace

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






def run_retinaface_detector(
    model,
    image_bgr,
    threshold: float,
    allow_upscaling: bool,
) -> list[tuple[float, float, float, float, float]]:
    """Run the serengil/retinaface detector and return [ymin, xmin, ymax, xmax, score] tuples."""
    faces = RetinaFace.detect_faces(
        img_path=image_bgr,
        threshold=threshold,
        model=model,
        allow_upscaling=allow_upscaling,
    )

    detections: list[tuple[float, float, float, float, float]] = []
    if not isinstance(faces, dict):
        return detections

    for face in faces.values():
        bbox = face.get("facial_area")
        score = float(face.get("score", 0.0))
        if not bbox or len(bbox) != 4 or score < threshold:
            continue

        x_min, y_min, x_max, y_max = map(float, bbox)
        if x_max <= x_min or y_max <= y_min:
            continue

        detections.append((y_min, x_min, y_max, x_max, score))

    return detections


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate RetinaFace detections and save them into a CSV file",
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
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.4,
        help="Detection score threshold"
    )
    parser.add_argument(
        "--allow-upscaling",
        action="store_true",
        help="Allow RetinaFace to upscale smaller images during preprocessing"
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

    # key = cv2.waitKey(1) & 0xFF
    # return key == 27




if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Setup RetinaFace detector (serengil/retinaface reference implementation)
    print("Loading RetinaFace model (serengil/retinaface)...")
    retinaface_model = RetinaFace.build_model()

    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_and_sort_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")
    print("Press ESC at any time to stop processing early.")

    current_idx = 0
    threshold = args.threshold

    #  create file next to our csv_path - if it was train.csv create train_new.csv
    new_csv_path = csv_path.with_name(csv_path.stem + "_new" + csv_path.suffix)
    csv_file = new_csv_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_path", "x1", "y1", "w", "h"])
    print(f"Writing filtered detections to {new_csv_path}")


    WINDOW = "RetinaFace Image Prep"
    # cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    processed_images: set[str] = set()
    
    try:
        with tqdm(total=len(image_paths), desc="Processing images", unit="img") as progress:
            while len(processed_images) < len(image_paths):
                # print(current_idx)
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

                detections = run_retinaface_detector(
                    retinaface_model,
                    img,
                    threshold,
                    allow_upscaling=args.allow_upscaling,
                )
                ih, iw, _ = img.shape
                
                
                if detections:
                                       
                    for ymin, xmin, ymax, xmax, score in detections:
                        x1 = max(0, min(iw - 1, int(round(xmin))))
                        y1 = max(0, min(ih - 1, int(round(ymin))))
                        x2 = max(0, min(iw - 1, int(round(xmax))))
                        y2 = max(0, min(ih - 1, int(round(ymax))))

                        width = max(0, x2 - x1)
                        height = max(0, y2 - y1)

                        if width == 0 or height == 0:
                            continue

                        # shrink height by 10% while keeping bottom edge fixed
                        reduced_height = max(1, int(round(height * 0.9)))
                        y1 = max(0, y2 - reduced_height)
                        height = y2 - y1
                        if width == 0 or height == 0:
                            continue

                        # Expand width symmetrically until aspect ratio ~1
                        target_width = height
                        if width < target_width:
                            pad = target_width - width
                            left_pad = pad // 2
                            right_pad = pad - left_pad
                            x1 = max(0, x1 - left_pad)
                            x2 = min(iw - 1, x2 + right_pad)
                            width = x2 - x1
                            if width <= 0:
                                continue

                        if width == 0 or height == 0:
                            continue

                        csv_writer.writerow([image_path, x1, y1, width, height])
                        cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)

                        cv2.putText(
                            img, f"{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )
 
                    cv2.putText(
                        img, f"{image_path}", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA
                    )
                    # cv2.imshow(WINDOW, img)
                    # key = cv2.waitKey(0) & 0xFF

                    # if key == 27 or key == ord('q'):  # ESC or 'q'
                    #     break                

                if esc_pressed():
                    print("ESC detected, stopping image processing loop.")
                    break

                current_idx = (current_idx + 1) % len(image_paths)
    finally:
        csv_file.close()

    print("done")
    cv2.destroyAllWindows()
    sys.exit(0)
