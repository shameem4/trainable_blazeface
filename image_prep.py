import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2
import mediapipe as mp
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

from retinaface import RetinaFace


MIN_AREA_RATIO = 0.005  # Faces smaller than ~0.5% of the frame are likely too tiny
MAX_AREA_RATIO = 0.4    # Faces occupying >40% may be cropped too tightly


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

def assess_detection_quality(
    img: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, float | list[str]]:
    """Compute heuristics (size ratio + blur) for a detected face."""
    ih, iw, _ = img.shape
    x, y, w, h = bbox
    frame_area = max(1, iw * ih)
    area_ratio = max(0.0, (w * h) / frame_area)

    warnings: list[str] = []
    if area_ratio < MIN_AREA_RATIO:
        warnings.append("too small")
    elif area_ratio > MAX_AREA_RATIO:
        warnings.append("overfill")



    return {
        "area_ratio": area_ratio,
        "warnings": warnings
    }

def pad(x1: int, y1: int, x2: int, y2: int, width: int, height: int,
        img_w: int, img_h: int, pad_ratio: float = 0.1
) -> tuple[int, int, int, int, int, int]:
    """Symmetrically expand bbox width while keeping height intact."""

    y1 = y1 + int(height * 0.1)
    height = max(1, y2 - y1)

    cx = (x1 + x2) / 2.0
    padded_width = width * (1.0 + pad_ratio * 2)
    half_width = padded_width / 2.0

    new_x1 = int(round(cx - half_width))
    new_x2 = int(round(cx + half_width))
    new_y1 = y1
    new_y2 = y2

    new_x1 = max(0, new_x1)
    new_x2 = min(img_w, new_x2)
    new_y1 = max(0, new_y1)
    new_y2 = min(img_h, new_y2)

    new_width = new_x2 - new_x1
    if new_width <= 0:
        new_x1 = max(0, int(x1))
        new_x2 = min(img_w, new_x1 + 1)
        new_width = new_x2 - new_x1

    return new_x1, new_y1, new_x2, new_y2, new_width, new_y2 - new_y1





def find_pose(landmarks, image_size, focal_length=None):
    """
    landmarks: dict with keys
        ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        Each value is (x, y) in pixel coordinates.
    image_size: (width, height)
    focal_length: optional; if None, uses width as fx = fy.
    
    Returns:
        rvec, tvec, (yaw, pitch, roll) in degrees
    """

    w, h = image_size
    if focal_length is None:
        focal_length = w  # crude but often usable approximation

    # --- 3D model points in some face model coordinate system (in mm-ish) ---
    # Coordinate system: origin roughly at nose tip.
    # +X: to the right, +Y: down, +Z: forward (toward camera).
    model_points = np.array([
        [-30.0,  -30.0,   30.0],  # left_eye
        [ 30.0,  -30.0,   30.0],  # right_eye
        [  0.0,    0.0,    0.0],  # nose
        [-40.0,   40.0,   30.0],  # mouth_left
        [ 40.0,   40.0,   30.0],  # mouth_right
    ], dtype=np.float32)

    # --- 2D image points from landmarks dict ---
    image_points = np.array([
        landmarks["left_eye"],
        landmarks["right_eye"],
        landmarks["nose"],
        landmarks["mouth_left"],
        landmarks["mouth_right"],
    ], dtype=np.float32)

    # --- Camera intrinsics (simple pinhole) ---
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1.0]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # assume no distortion

    # --- Solve PnP ---
    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        raise RuntimeError("solvePnP failed to find a valid pose.")

    # --- Convert rvec to rotation matrix ---
    R, _ = cv2.Rodrigues(rvec)

    # --- Get Euler angles (yaw, pitch, roll) from rotation matrix ---
    # Using cv2.decomposeProjectionMatrix to avoid hand-deriving
    proj_matrix = np.hstack((R, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = euler_angles.flatten()  # OpenCV order: pitch, yaw, roll (deg)

    return rvec, tvec, (float(yaw), float(pitch), float(roll))




if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_and_sort_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")

    current_idx = 0
    processed_count = 0
    window_name = "MediaPipe vs CSV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    threshold = 0.95

    # Create file next to our csv_path - if it was train.csv create train_new.csv
    new_csv_path = csv_path.with_name(csv_path.stem + "_new" + csv_path.suffix)

    # Initialize the new CSV file with headers
    new_df = pd.DataFrame(columns=['image_path', 'x1', 'y1', 'w', 'h', 'width', 'height'])
    new_df.to_csv(new_csv_path, index=False)

    while current_idx < len(image_paths):
        image_path = image_paths[current_idx]

        
        full_image_path = SCRIPT_DIR / args.data_root / image_path

        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            current_idx += 1
            continue

        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"Warning: Failed to load image: {full_image_path}")
            current_idx += 1
            continue
        # print(f"Processing {image_path} ({current_idx + 1}/{len(image_paths)})")
        faces = RetinaFace.detect_faces(str(full_image_path))


        detections_found = False
        ih, iw, _ = img.shape
        if faces:
            for key in faces.keys():
                detection = faces[key]
                score = detection['score']
                if score < threshold:
                    continue
                detections_found = True
                x1, y1, x2, y2 = detection['facial_area']
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(iw, int(x2))
                y2 = min(ih, int(y2))
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)

                quality = assess_detection_quality(img, (x1, y1, width, height))
                warn_text = ", ".join(quality["warnings"]) if quality["warnings"] else "OK"
                print(
                    f"{image_path} - score {score:.2f}, area {quality['area_ratio']:.3f}"
                )

                x1, y1, x2, y2, width, height = pad(
                    x1, y1, x2, y2, width, height, iw, ih
                )

                if quality["warnings"]:
                    continue
                # color = (10, 255, 10) if not quality["warnings"] else (10, 10, 255)
                # cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), color, 2)


                # Add row to new_csv_path - create new row with image_path, x1, y1, width, height
                new_row = pd.DataFrame([{
                    'image_path': image_path,
                    'x1': x1,
                    'y1': y1,
                    'w': width,
                    'h': height,
                    'width': iw,
                    'height': ih
                }])
                new_row.to_csv(new_csv_path, mode='a', header=False, index=False)
                processed_count += 1



                # # draw landmarks on image 
                # for lm_name, lm_coord in detection['landmarks'].items():
                #     lm_x = int(lm_coord[0])
                #     lm_y = int(lm_coord[1])
                #     cv2.circle(img, (lm_x, lm_y), 2, (255, 0, 0), -1)


                # rvec, tvec, (yaw, pitch, roll) = find_pose(detection['landmarks'], img.shape[:2])
                
                # label_text = f"{yaw:.3f}"
                # (text_width, text_height), baseline = cv2.getTextSize(
                #     label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                # )
                # y_tab = y1 - 10 if y1 - 10 > 10 else y1 + text_height
                # cv2.rectangle(
                #     img,
                #     (x1, y_tab - text_height - baseline),
                #     (x1 + text_width + 5, y_tab + baseline),
                #     color,
                #     -1,
                # )
                # cv2.putText(
                #     img,
                #     label_text,
                #     (x1, y_tab),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     (255, 255, 255),
                #     2,
                # )
                # cv2.putText(
                #     img,
                #     warn_text,
                #     (x1, y_tab + text_height + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.5,
                #     color,
                #     1,
                # )


        if not detections_found:
            # print(f"No MediaPipe detections >=0.7 for {image_path}, skipping.")
            current_idx += 1
            continue
        
        # Move to next image after processing detections
        current_idx += 1
        
        # cv2.imshow(window_name, img)

        # key = cv2.waitKey(0) & 0xFF
        # if key == 27 or key == ord("q"):
        #     break
        # elif key == ord("a") or key == 81:
        #     current_idx = (current_idx - 1) % len(image_paths)
        #     # print(f"Previous: {current_idx + 1}/{len(image_paths)}")
        # else:
        #     current_idx = (current_idx + 1) % len(image_paths)
        #     # print(f"Next: {current_idx + 1}/{len(image_paths)}")

    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} detections from {len(image_paths)} images")
    print(f"Output saved to: {new_csv_path}")
    
    cv2.destroyAllWindows()
    sys.exit(0)
