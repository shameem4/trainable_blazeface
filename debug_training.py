import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from dataloader import CSVDetectorDataset
from blazeface import BlazeFace
from blazebase import anchor_options, generate_reference_anchors, load_mediapipe_weights
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu()
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
        f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
    )


def box_iou(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one GT box and N predicted boxes."""
    gt = gt.unsqueeze(0)  # [1, 4]
    ymin = torch.maximum(gt[:, 0], pred[:, 0])
    xmin = torch.maximum(gt[:, 1], pred[:, 1])
    ymax = torch.minimum(gt[:, 2], pred[:, 2])
    xmax = torch.minimum(gt[:, 3], pred[:, 3])

    inter_h = torch.clamp(ymax - ymin, min=0)
    inter_w = torch.clamp(xmax - xmin, min=0)
    intersection = inter_h * inter_w

    gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union = gt_area + pred_area - intersection + 1e-6
    return (intersection / union).squeeze(0)


def compute_resize_metadata(
    orig_h: int,
    orig_w: int,
    target_size: Tuple[int, int]
) -> Tuple[float, int, int]:
    """Recreate resize/pad parameters used during preprocessing."""
    target_h, target_w = target_size

    if orig_h >= orig_w:
        scale = target_h / orig_h
        new_h = target_h
        new_w = int(round(orig_w * scale))
    else:
        scale = target_w / orig_w
        new_w = target_w
        new_h = int(round(orig_h * scale))

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2

    return scale, pad_top, pad_left


def map_preprocessed_boxes_to_original(
    boxes_xyxy: np.ndarray,
    orig_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    scale: float,
    pad_top: int,
    pad_left: int
) -> np.ndarray:
    """Map boxes from 128x128 preprocessed space back to original resolution."""
    if boxes_xyxy.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    target_h, target_w = target_size
    orig_h, orig_w = orig_shape
    boxes = np.array(boxes_xyxy, dtype=np.float32, copy=True)

    boxes[:, [0, 2]] *= target_w  # x coords
    boxes[:, [1, 3]] *= target_h  # y coords
    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

    return boxes


def draw_box(
    image: np.ndarray,
    box: np.ndarray,
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    thickness: int = 2
) -> None:
    """Draw a single rectangle (with optional label) on the debug image."""
    x1, y1, x2, y2 = box.astype(int).tolist()
    x1 = int(np.clip(x1, 0, image.shape[1] - 1))
    x2 = int(np.clip(x2, 0, image.shape[1] - 1))
    y1 = int(np.clip(y1, 0, image.shape[0] - 1))
    y2 = int(np.clip(y2, 0, image.shape[0] - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, 1)
        text_y = max(y1 - baseline, text_h + 2)
        cv2.rectangle(
            image,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline // 2),
            color,
            thickness=cv2.FILLED
        )
        cv2.putText(
            image,
            label,
            (x1, text_y - 2),
            font,
            scale,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA
        )


def create_debug_visualization(
    dataset: CSVDetectorDataset,
    sample_idx: int,
    decoded_boxes: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    output_root: Path,
    top_k: int = 5
) -> Path:
    """Create a debug overlay showing GT/padded GT/model predictions."""
    image_path, gt_boxes_xywh = dataset.get_sample_annotations(sample_idx)
    orig_image = dataset._load_image(image_path)
    orig_h, orig_w = orig_image.shape[:2]

    if len(gt_boxes_xywh) > 0:
        x1 = gt_boxes_xywh[:, 0]
        y1 = gt_boxes_xywh[:, 1]
        w = gt_boxes_xywh[:, 2]
        h = gt_boxes_xywh[:, 3]

        gt_box_orig = np.stack([x1, y1, x1 + w, y1 + h], axis=1).astype(np.float32)

        ymin = y1 / orig_h
        xmin = x1 / orig_w
        ymax = (y1 + h) / orig_h
        xmax = (x1 + w) / orig_w
        gt_norm = np.stack([ymin, xmin, ymax, xmax], axis=1).astype(np.float32)
    else:
        gt_box_orig = np.zeros((0, 4), dtype=np.float32)
        gt_norm = np.zeros((0, 4), dtype=np.float32)

    _, resized_boxes_norm = dataset._resize_and_pad(orig_image, gt_norm.copy())
    gt_resized_xy = resized_boxes_norm[:, [1, 0, 3, 2]] if len(resized_boxes_norm) > 0 else np.zeros((0, 4), dtype=np.float32)

    scale, pad_top, pad_left = compute_resize_metadata(orig_h, orig_w, dataset.target_size)
    gt_resized_on_orig = map_preprocessed_boxes_to_original(
        gt_resized_xy,
        (orig_h, orig_w),
        dataset.target_size,
        scale,
        pad_top,
        pad_left
    )

    debug_image = cv2.cvtColor(orig_image.copy(), cv2.COLOR_RGB2BGR)

    for box in gt_box_orig:
        draw_box(debug_image, box, (0, 255, 0), "GT original")

    # for box in gt_resized_on_orig:
    #     draw_box(debug_image, box, (0, 165, 255), "GT resized->orig")

    decoded_np = decoded_boxes.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()
    top_scores_np = top_scores.detach().cpu().numpy()
    top_k = min(len(top_indices_np), top_k)
    pred_boxes = decoded_np[top_indices_np[:top_k]]
    pred_boxes_on_orig = map_preprocessed_boxes_to_original(
        pred_boxes,
        (orig_h, orig_w),
        dataset.target_size,
        scale,
        pad_top,
        pad_left
    )

    for rank, (box, score, anchor_idx) in enumerate(
        zip(pred_boxes_on_orig, top_scores_np[:top_k], top_indices_np[:top_k])
    ):
        label = f"Pred {rank} #{anchor_idx} {score:.2f}"
        draw_box(debug_image, box, (0, 0, 255), label)

    debug_dir = output_root / "debug_images"
    debug_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    debug_path = debug_dir / f"sample_{sample_idx:04d}_{image_stem}.png"
    cv2.imwrite(str(debug_path), debug_image)
    return debug_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug BlazeFace training sample (single image end-to-end)"
    )
    parser.add_argument("--csv", type=str, default="data/splits/train_new.csv")
    parser.add_argument("--data-root", type=str, default="data/raw/blazeface")
    parser.add_argument("--weights", type=str, default="model_weights/blazeface.pth")
    parser.add_argument("--index", type=int, default=2, help="Sample index to inspect")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dataset = CSVDetectorDataset(
        csv_path=str(csv_path),
        root_dir=args.data_root,
        target_size=(128, 128),
        augment=False
    )

    if args.index >= len(dataset):
        raise IndexError(f"Index {args.index} is out of range (dataset has {len(dataset)} samples).")

    sample = dataset[args.index]
    image = sample["image"].unsqueeze(0)  # [1, 3, H, W]
    anchor_targets = sample["anchor_targets"]

    print(f"Loaded sample {args.index} from {csv_path}")
    describe_tensor("image", image)

    positives = anchor_targets[:, 0].sum().item()
    print(f"Positive anchors: {positives}/896")
    if positives > 0:
        first_pos = anchor_targets[anchor_targets[:, 0] == 1][:3]
        print("First positive targets (class, ymin, xmin, ymax, xmax):")
        print(first_pos)

    # Load BlazeFace model with MediaPipe weights
    model = BlazeFace()
    load_mediapipe_weights(model, args.weights, strict=False)
    model.eval()
    model.generate_anchors(anchor_options)

    with torch.no_grad():
        raw_boxes, raw_scores = model.get_training_outputs(image)

    class_predictions = torch.sigmoid(raw_scores).squeeze(0)  # [896, 1]
    anchor_predictions = raw_boxes.squeeze(0)[..., :4]        # [896, 4]

    describe_tensor("class_predictions", class_predictions)
    describe_tensor("anchor_predictions", anchor_predictions)

    top_scores, top_indices = torch.topk(class_predictions.squeeze(-1), k=10)
    print("Top-10 anchor scores:")
    for score, idx in zip(top_scores, top_indices):
        print(f"  idx={idx.item():4d} score={score.item():.4f}")

    loss_fn = BlazeFaceDetectionLoss()
    reference_anchors, _, _ = generate_reference_anchors()
    decoded_boxes = loss_fn.decode_boxes(
        anchor_predictions.unsqueeze(0),
        reference_anchors
    ).squeeze(0)

    gt_boxes = anchor_targets[:, 1:]
    positive_mask = anchor_targets[:, 0] > 0

    if positive_mask.any():
        mean_iou = compute_mean_iou(
            decoded_boxes[positive_mask],
            gt_boxes[positive_mask]
        )
        print(f"Mean IoU on positive anchors: {mean_iou.item():.4f}")

        # Inspect the first positive anchor
        first_gt = gt_boxes[positive_mask][0]
        print(f"GT box (first positive): {first_gt}")

        positive_indices = torch.nonzero(positive_mask).squeeze(1)
        print("Positive anchor indices:", positive_indices.tolist())

        pos_boxes = decoded_boxes[positive_indices]
        gt_iou = box_iou(first_gt, pos_boxes)
        for idx, (box, iou) in zip(positive_indices.tolist(), zip(pos_boxes.tolist(), gt_iou.tolist())):
            print(f"  Anchor {idx}: box={box}, IoU={iou:.4f}")

        # IoU of top scoring anchors with GT
        top_iou = box_iou(first_gt, decoded_boxes[top_indices])
        print("IoU of top scoring anchors w.r.t first GT:")
        for idx, score, box, iou in zip(
            top_indices.tolist(),
            top_scores.tolist(),
            decoded_boxes[top_indices].tolist(),
            top_iou.tolist()
        ):
            print(f"  idx={idx:4d} score={score:.4f} box={box} IoU={iou:.4f}")
    else:
        print("No positive anchors in this sample.")

    debug_path = create_debug_visualization(
        dataset=dataset,
        sample_idx=args.index,
        decoded_boxes=decoded_boxes,
        top_indices=top_indices,
        top_scores=top_scores,
        output_root=Path("logs")
    )
    print(f"Saved debug visualization to {debug_path}")


if __name__ == "__main__":
    main()
