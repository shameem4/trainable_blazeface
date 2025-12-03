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
import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

from blazeface import BlazeFace


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


def draw_ground_truth_boxes(
    img: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    label: str = "GT"
) -> None:
    """Draw ground truth bounding boxes from CSV annotations.

    Args:
        img: Image to draw on (modified in place)
        boxes: List of boxes in format (x1, y1, w, h)
        color: BGR color tuple (default: blue for ground truth)
        thickness: Line thickness
        label: Label text to show above boxes
    """
    for i, (x1, y1, w, h) in enumerate(boxes):
        # Convert (x1, y1, w, h) to (x1, y1, x2, y2)
        x2 = x1 + w
        y2 = y1 + h

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Draw label above box
        label_text = f"{label} {i+1}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = y1 - 5 if y1 > label_h + 5 else y1 + label_h + 5
        cv2.putText(
            img, label_text, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def draw_detections(
    img: np.ndarray,
    detections: torch.Tensor | np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = "Det"
) -> None:
    """Draw bounding boxes and confidence scores from detections.

    Args:
        img: Image to draw on (modified in place)
        detections: Detection tensor [N, 5] with format [ymin, xmin, ymax, xmax, score]
        color: BGR color tuple (default: green for detections)
        thickness: Line thickness
        label: Label prefix for detections
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

        # Get confidence score (index 4)
        score = detections[i, 4] if detections.shape[1] > 4 else 0.0

        # Draw axis-aligned bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Draw confidence score and label above the box
        label_text = f"{label} {i+1}: {score:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Position label above box, or inside if at top edge
        label_y = ymin - 5 if ymin > label_h + 5 else ymin + label_h + 5
        cv2.putText(
            img, label_text, (xmin, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )


def draw_info_text(
    img: np.ndarray,
    image_idx: int,
    total_images: int,
    num_gt_boxes: int,
    num_detections: int,
    image_path: str
) -> None:
    """Draw information text at the top of the image."""
    # Background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # Text lines
    text_lines = [
        f"Image {image_idx + 1}/{total_images}: {Path(image_path).name}",
        f"Ground Truth: {num_gt_boxes} boxes | Detections: {num_detections} boxes",
        "Controls: A/D or Arrow keys to navigate | Q/ESC to quit"
    ]

    y_offset = 20
    for line in text_lines:
        cv2.putText(
            img, line, (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        y_offset += 20


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
        default=0.3,
        help="Detection threshold (overrides model default)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start from this image index"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SCRIPT_DIR = Path(__file__).parent

    # Setup device
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {gpu}")
    torch.set_grad_enabled(False)

    # Load model
    print(f"Loading weights: {args.weights}")
    detector = load_model(args.weights, gpu)

    # Override detection threshold if specified
    if args.threshold is not None:
        detector.min_score_thresh = args.threshold
        print(f"Detection threshold: {args.threshold}")

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

        # Convert detections to numpy if needed
        if isinstance(detections, torch.Tensor):
            detections_np = detections.cpu().numpy()
        else:
            detections_np = detections

        # Create display image
        display_img = img.copy()

        # Draw ground truth boxes (blue)
        draw_ground_truth_boxes(display_img, gt_boxes, color=(255, 0, 0), thickness=2, label="GT")

        # Draw detections (green)
        if len(detections_np) > 0:
            draw_detections(display_img, detections_np, color=(0, 255, 0), thickness=2, label="Det")

        # Draw info text
        draw_info_text(
            display_img,
            current_idx,
            len(image_paths),
            len(gt_boxes),
            len(detections_np),
            image_path
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
