"""
Image demo for BlazeFace detection.
Loads images from train.csv and compares detector output with ground truth annotations.

Navigation:
- 'a' or Left Arrow: Previous image
- 'd' or Right Arrow: Next image
- 'q' or ESC: Quit

Supports loading:
- MediaPipe weights: blazeface.pth (raw state_dict)
- Retrained checkpoints: *.ckpt (dict with 'model_state_dict' key)
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

from utils import model_utils, drawing, metrics, config


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
        "--weights", "-w",
        type=str,
        default="model_weights/blazeface.pth",
        # default="checkpoints/BlazeFace_best.pth",
        help="Path to weights file (.pth for MediaPipe, .ckpt for retrained)"
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
        default=0.5,
        help="Detection threshold (overrides model default)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start from this image index"
    )
    detection_mode = parser.add_mutually_exclusive_group()
    detection_mode.add_argument(
        "--detection-only",
        dest="detection_only",
        action="store_true",
        help="Show only detections without ground truth comparison (disables IoU matching)"
    )
    detection_mode.add_argument(
        "--no-detection-only",
        dest="detection_only",
        action="store_false",
        help="Enable ground truth overlays and IoU matching"
    )
    parser.set_defaults(detection_only=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Setup device
    gpu = model_utils.setup_device()

    # Load model with threshold
    print(f"Loading weights: {args.weights}")
    detector = model_utils.load_model(args.weights, gpu, threshold=args.threshold)

    print("Model loaded")

    # Load CSV and sort by image path
    csv_path = SCRIPT_DIR / args.csv
    print(f"Loading CSV: {csv_path}")
    image_paths, image_to_boxes = load_and_sort_csv(str(csv_path))
    print(f"Loaded {len(image_paths)} unique images with annotations")

    # Setup window
    WINDOW = "BlazeFace Image Demo"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Navigation state
    current_idx = min(args.start_idx, len(image_paths) - 1)

    print("\nControls:")
    print("  A / Left Arrow  - Previous image")
    print("  D / Right Arrow - Next image")
    print("  Q / ESC         - Quit")
    print()

    while True:
        # Get current image info
        image_path = image_paths[current_idx]
        gt_boxes = image_to_boxes[image_path]

        # Construct full path
        full_image_path = SCRIPT_DIR / args.data_root / image_path

        # Load image
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        img = cv2.imread(str(full_image_path))
        if img is None:
            print(f"Warning: Failed to load image: {full_image_path}")
            current_idx = (current_idx + 1) % len(image_paths)
            continue

        # Convert BGR to RGB for detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run detection
        detections = detector.process(img_rgb)

        # Convert detections to numpy if needed (drawing functions handle this too)
        detections_np = detections.cpu().numpy() if hasattr(detections, 'cpu') else detections

        # Create display image
        display_img = img.copy()

        if args.detection_only:
            # Detection-only mode: show all detections, no ground truth
            if len(detections_np) > 0:
                drawing.draw_detections(
                    display_img, detections_np, indices=None,
                    color=(0, 255, 0), thickness=2, label="Det"
                )

            # Draw info text (detection-only mode)
            drawing.draw_info_text(
                display_img,
                current_idx,
                len(image_paths),
                num_gt_boxes=0,
                num_detections=len(detections_np),
                num_matched=0,
                avg_iou=0.0,
                image_path=image_path,
                detection_only=True
            )
        else:
            # Comparison mode: match detections to ground truth
            matched_indices, matched_ious = metrics.match_detections_to_ground_truth(
                gt_boxes, detections_np, iou_threshold=0.3
            )

            # Calculate stats
            num_matched = sum(1 for idx in matched_indices if idx != -1)
            avg_iou = np.mean([iou for iou in matched_ious if iou > 0]) if num_matched > 0 else 0.0

            # Draw ground truth boxes (blue) with IoU values
            drawing.draw_ground_truth_boxes(
                display_img, gt_boxes, ious=matched_ious,
                color=(255, 0, 0), thickness=2, label="GT"
            )

            # Draw only matched detections (green)
            if len(detections_np) > 0:
                drawing.draw_detections(
                    display_img, detections_np, indices=matched_indices,
                    color=(0, 255, 0), thickness=2, label="Det"
                )

            # Draw info text (comparison mode)
            drawing.draw_info_text(
                display_img,
                current_idx,
                len(image_paths),
                len(gt_boxes),
                len(detections_np),
                num_matched,
                avg_iou,
                image_path,
                detection_only=False
            )

        # Display
        cv2.imshow(WINDOW, display_img)

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
