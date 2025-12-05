import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from dataloader import CSVDetectorDataset, encode_boxes_to_anchors, flatten_anchor_targets
from blazeface import BlazeFace
from blazedetector import BlazeDetector
from blazebase import anchor_options, generate_reference_anchors, load_mediapipe_weights
from loss_functions import BlazeFaceDetectionLoss, compute_mean_iou
from utils import model_utils

LOSS_DEBUG_KWARGS = {
    "use_focal_loss": True,
    "positive_classification_weight": 70.0
}


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


def aligned_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute IoU for aligned [N,4] tensors."""
    ymin = torch.maximum(pred[:, 0], target[:, 0])
    xmin = torch.maximum(pred[:, 1], target[:, 1])
    ymax = torch.minimum(pred[:, 2], target[:, 2])
    xmax = torch.minimum(pred[:, 3], target[:, 3])

    inter_h = torch.clamp(ymax - ymin, min=0)
    inter_w = torch.clamp(xmax - xmin, min=0)
    intersection = inter_h * inter_w

    pred_area = torch.clamp(pred[:, 2] - pred[:, 0], min=0) * torch.clamp(pred[:, 3] - pred[:, 1], min=0)
    target_area = torch.clamp(target[:, 2] - target[:, 0], min=0) * torch.clamp(target[:, 3] - target[:, 1], min=0)
    union = pred_area + target_area - intersection + 1e-6
    return intersection / union


def run_anchor_unit_tests() -> None:
    print("\nRunning anchor sanity checks...")
    # Single box centered perfectly
    box = np.array([[0.4, 0.4, 0.6, 0.6]], dtype=np.float32)
    small, big = encode_boxes_to_anchors(box, input_size=128)
    targets = flatten_anchor_targets(small, big)
    positives = np.where(targets[:, 0] == 1)[0]
    if positives.size == 0:
        raise AssertionError("No anchors assigned to centered box")
    np.testing.assert_allclose(targets[positives[0], 1:], box[0], atol=1e-3)
    print(f"  PASS single box assigned to anchor #{positives[0]} with coords {targets[positives[0], 1:]}")

    # Multiple boxes shouldn't overwrite each other
    multi_boxes = np.array([
        [0.1, 0.1, 0.2, 0.2],
        [0.7, 0.7, 0.85, 0.85]
    ], dtype=np.float32)
    small, big = encode_boxes_to_anchors(multi_boxes, input_size=128)
    targets = flatten_anchor_targets(small, big)
    assigned = np.where(targets[:, 0] == 1)[0]
    if assigned.size < 2:
        raise AssertionError("Not all boxes were assigned to anchors")
    print(f"  PASS multiple boxes mapped to anchors {assigned.tolist()}")


def run_decode_unit_test(loss_fn: BlazeFaceDetectionLoss) -> None:
    print("Running decode sanity check...")
    reference_anchors, _, _ = generate_reference_anchors()
    gt = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    preds = torch.zeros((1, 1, 4))
    preds[..., 0] = (gt[0, 1] + gt[0, 3]) / 2 * loss_fn.scale - reference_anchors[0, 0]
    preds[..., 1] = (gt[0, 0] + gt[0, 2]) / 2 * loss_fn.scale - reference_anchors[0, 1]
    preds[..., 2] = (gt[0, 3] - gt[0, 1]) * loss_fn.scale
    preds[..., 3] = (gt[0, 2] - gt[0, 0]) * loss_fn.scale
    decoded = loss_fn.decode_boxes(preds, reference_anchors).squeeze(0)
    torch.testing.assert_close(decoded[0], gt[0], atol=1e-3)
    decoded_xyxy = convert_ymin_xmin_to_xyxy(decoded[0].cpu().numpy())
    expected_xyxy = convert_ymin_xmin_to_xyxy(gt[0].cpu().numpy())
    torch.testing.assert_close(torch.from_numpy(decoded_xyxy), torch.from_numpy(expected_xyxy), atol=1e-3)
    print("  PASS decode matches ground truth for synthetic box")


def run_csv_encode_decode_test(
    dataset: CSVDetectorDataset,
    max_samples: int = 3
) -> None:
    """Ensure encode/decode math is consistent with CSV-derived GT boxes."""
    print("\nRunning CSV encode/decode consistency test...")
    reference_anchors, _, _ = generate_reference_anchors()
    loss_fn = BlazeFaceDetectionLoss(**LOSS_DEBUG_KWARGS)
    sample_indices = list(range(min(max_samples, len(dataset))))

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        anchor_targets = sample["anchor_targets"]
        positive_mask = anchor_targets[:, 0] == 1
        if not bool(positive_mask.any()):
            print(f"  Sample {sample_idx}: no positives, skipping")
            continue

        pos_indices = torch.nonzero(positive_mask).squeeze(1)
        true_boxes = anchor_targets[pos_indices, 1:]
        anchor_predictions = torch.zeros((reference_anchors.shape[0], 4), dtype=torch.float32)

        for anchor_idx, true_box in zip(pos_indices.tolist(), true_boxes):
            anchor = reference_anchors[anchor_idx]
            y_min, x_min, y_max, x_max = true_box.tolist()
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            anchor_w = anchor[2].item()
            anchor_h = anchor[3].item()

            anchor_predictions[anchor_idx, 0] = ((x_center - anchor[0].item()) / anchor_w) * loss_fn.scale
            anchor_predictions[anchor_idx, 1] = ((y_center - anchor[1].item()) / anchor_h) * loss_fn.scale
            anchor_predictions[anchor_idx, 2] = (width / anchor_w) * loss_fn.scale
            anchor_predictions[anchor_idx, 3] = (height / anchor_h) * loss_fn.scale

        decoded = loss_fn.decode_boxes(anchor_predictions.unsqueeze(0), reference_anchors).squeeze(0)
        decoded_pos = decoded[pos_indices]
        max_error = (decoded_pos - true_boxes).abs().max().item()
        mean_iou = compute_mean_iou(decoded_pos, true_boxes).item()
        print(
            f"  Sample {sample_idx}: positives={len(pos_indices)}, "
            f"max_abs_error={max_error:.6f}, mean_iou={mean_iou:.4f}"
        )


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


def convert_ymin_xmin_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [ymin, xmin, ymax, xmax] boxes to [xmin, ymin, xmax, ymax]."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    return boxes[:, [1, 0, 3, 2]]


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


def _select_top_indices(
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    top_k: int
) -> tuple[list[int], list[float]]:
    """Return up to top_k anchor indices prioritizing positives."""
    selected_indices: list[int] = []
    selected_scores: list[float] = []

    for anchor_idx in top_indices.detach().cpu().numpy():
        if anchor_targets[anchor_idx, 0] == 1:
            selected_indices.append(anchor_idx)
            selected_scores.append(class_predictions[anchor_idx].item())

    if not selected_indices:
        selected_indices = top_indices.detach().cpu().numpy().tolist()
        selected_scores = top_scores.detach().cpu().numpy().tolist()

    selected_indices = selected_indices[:top_k]
    selected_scores = selected_scores[:top_k]
    return selected_indices, selected_scores


def analyze_scoring_process(
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    decoded_boxes: torch.Tensor,
    top_k: Tuple[int, int] = (10, 50)
) -> None:
    """Trace how classification scores align with GT assignments."""
    scores = class_predictions.detach().cpu().squeeze(-1)
    decoded = decoded_boxes.detach().cpu()
    targets = anchor_targets.detach().cpu()

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    highest_idx = sorted_indices[0].item()

    pos_mask = targets[:, 0] == 1
    positive_indices = torch.nonzero(pos_mask).squeeze(1)
    pos_count = int(positive_indices.numel())

    print("\nScoring diagnostics:")
    if pos_count == 0:
        print("  No positive anchors available for scoring analysis.")
        return

    pos_scores = scores[positive_indices]
    pos_boxes = targets[positive_indices, 1:]
    decoded_pos = decoded[positive_indices]
    pos_iou = aligned_iou(decoded_pos, pos_boxes)

    mean_score = pos_scores.mean().item()
    mean_iou = pos_iou.mean().item()
    corr = float("nan")
    if pos_scores.numel() > 1:
        stacked = torch.stack([pos_scores, pos_iou])
        corr = torch.corrcoef(stacked)[0, 1].item()

    best_iou_idx = torch.argmax(pos_iou).item()
    best_iou_anchor = positive_indices[best_iou_idx].item()
    best_iou_score = pos_scores[best_iou_idx].item()
    best_iou_value = pos_iou[best_iou_idx].item()
    best_iou_rank = (sorted_indices == best_iou_anchor).nonzero(as_tuple=False)[0].item() + 1

    best_score_idx = torch.argmax(pos_scores).item()
    best_score_anchor = positive_indices[best_score_idx].item()
    best_score_value = pos_scores[best_score_idx].item()
    best_score_iou = pos_iou[best_score_idx].item()
    best_score_rank = (sorted_indices == best_score_anchor).nonzero(as_tuple=False)[0].item() + 1

    highest_iou = aligned_iou(decoded[highest_idx].unsqueeze(0), targets[highest_idx, 1:].unsqueeze(0)).item() \
        if targets[highest_idx, 0] == 1 else 0.0

    print(
        f"  positives={pos_count}, mean_score={mean_score:.3f}, "
        f"mean_iou={mean_iou:.3f}, score/iou corr={corr:.3f}"
    )
    print(
        f"  best IoU anchor #{best_iou_anchor} -> IoU={best_iou_value:.3f}, "
        f"score={best_iou_score:.3f}, score rank={best_iou_rank}"
    )
    print(
        f"  best score anchor #{best_score_anchor} -> score={best_score_value:.3f}, "
        f"IoU={best_score_iou:.3f}, score rank={best_score_rank}"
    )
    print(
        f"  global top score anchor #{highest_idx} "
        f"{'(positive)' if targets[highest_idx,0]==1 else '(background)'} "
        f"IoU={highest_iou:.3f}, score={sorted_scores[0].item():.3f}"
    )

    for k in top_k:
        window = min(k, len(sorted_indices))
        selected = sorted_indices[:window]
        positive_hits = int(targets[selected, 0].sum().item())
        print(f"  positives within top-{window} scores: {positive_hits}/{window}")


def create_debug_visualization(
    dataset: CSVDetectorDataset,
    sample_idx: int,
    decoded_boxes: torch.Tensor,
    top_indices: torch.Tensor,
    top_scores: torch.Tensor,
    anchor_targets: torch.Tensor,
    class_predictions: torch.Tensor,
    output_root: Path,
    top_k: int = 5,
    comparison_detector: Optional[BlazeFace] = None,
    comparison_label: str = "Det2"
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
    gt_resized_xy = convert_ymin_xmin_to_xyxy(resized_boxes_norm)

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

    comparison_np: Optional[np.ndarray] = None
    if comparison_detector is not None:
        comparison_input = np.ascontiguousarray(orig_image)
        try:
            detections = comparison_detector.process(comparison_input)
            if detections is not None and detections.numel() > 0:
                comparison_np = detections.detach().cpu().numpy()
                print(f"{comparison_label}: {len(comparison_np)} detections")
        except Exception as exc:  # pragma: no cover - debug helper
            print(f"Secondary detector failed on sample {sample_idx}: {exc}")

    for box in gt_box_orig:
        draw_box(debug_image, box, (0, 255, 0), "GT original")

    # for box in gt_resized_on_orig:
    #     draw_box(debug_image, box, (0, 165, 255), "GT resized->orig")

    decoded_np = decoded_boxes.detach().cpu().numpy()
    selected_indices, selected_scores = _select_top_indices(
        anchor_targets, class_predictions, top_indices, top_scores, top_k
    )
    pred_boxes = convert_ymin_xmin_to_xyxy(decoded_np[selected_indices])
    pred_boxes_on_orig = map_preprocessed_boxes_to_original(
        pred_boxes,
        (orig_h, orig_w),
        dataset.target_size,
        scale,
        pad_top,
        pad_left
    )

    for rank, (box, score, anchor_idx) in enumerate(
        zip(pred_boxes_on_orig, selected_scores, selected_indices)
    ):
        label = f"Pred {rank} #{anchor_idx} {score:.2f}"
        draw_box(debug_image, box, (0, 0, 255), label)

    if comparison_np is not None and comparison_np.size > 0:
        for det_idx, det in enumerate(comparison_np):
            comp_box = np.array([det[1], det[0], det[3], det[2]], dtype=np.float32)
            score = float(det[-1]) if det.shape[0] > 4 else 0.0
            draw_box(
                debug_image,
                comp_box,
                color=(0, 165, 255),
                label=f"{comparison_label} {det_idx} {score:.2f}"
            )

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
    parser.add_argument("--weights", type=str, default="runs/checkpoints/BlazeFace_best.pth")
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Comma-separated list of sample indices or single index to inspect"
    )
    parser.add_argument(
        "--compare-weights",
        type=str,
        default="model_weights/blazeface.pth",
        help="Optional path to secondary detector weights (.pth or .ckpt) for visual comparison"
    )
    parser.add_argument(
        "--compare-threshold",
        type=float,
        default=0.5,
        help="Detection threshold for the secondary detector"
    )
    parser.add_argument(
        "--compare-label",
        type=str,
        default="Det2",
        help="Label prefix for the secondary detector annotations"
    )
    args = parser.parse_args()

    run_anchor_unit_tests()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dataset = CSVDetectorDataset(
        csv_path=str(csv_path),
        root_dir=args.data_root,
        target_size=(128, 128),
        augment=False
    )

    run_csv_encode_decode_test(dataset)

    comparison_detector = None
    if args.compare_weights:
        device = model_utils.setup_device()
        comparison_detector = model_utils.load_model(
            args.compare_weights,
            device=device,
            threshold=args.compare_threshold
        )
        print(f"Loaded comparison detector from {args.compare_weights}")

    if args.index is None:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=min(10, len(dataset)), replace=False)
    else:
        indices = [int(idx.strip()) for idx in args.index.split(",") if idx.strip()]

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            raise IndexError(f"Index {idx} is out of range (dataset has {len(dataset)} samples).")

        sample = dataset[idx]
        image = sample["image"].unsqueeze(0)  # [1, 3, H, W]
        anchor_targets = sample["anchor_targets"]

        print(f"\nLoaded sample {idx} from {csv_path}")
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
        selected_indices, selected_scores = _select_top_indices(
            anchor_targets, class_predictions.squeeze(-1), top_indices, top_scores, top_k=5
        )
        print("Top-10 anchor scores:")
        for score, anchor_idx in zip(top_scores, top_indices):
            print(f"  idx={anchor_idx.item():4d} score={score.item():.4f}")

        loss_fn = BlazeFaceDetectionLoss(**LOSS_DEBUG_KWARGS)
        reference_anchors, _, _ = generate_reference_anchors()
        decoded_boxes = loss_fn.decode_boxes(
            anchor_predictions.unsqueeze(0),
            reference_anchors
        ).squeeze(0)

        analyze_scoring_process(
            anchor_targets=anchor_targets,
            class_predictions=class_predictions.squeeze(-1),
            decoded_boxes=decoded_boxes
        )

        gt_boxes = anchor_targets[:, 1:]
        positive_mask = anchor_targets[:, 0] > 0

        if positive_mask.any():
            mean_iou = compute_mean_iou(
                decoded_boxes[positive_mask],
                gt_boxes[positive_mask]
            )
            print(f"Mean IoU on positive anchors: {mean_iou.item():.4f}")

            first_gt = gt_boxes[positive_mask][0]
            print(f"GT box (first positive): {first_gt}")

            positive_indices = torch.nonzero(positive_mask).squeeze(1)
            print("Positive anchor indices:", positive_indices.tolist())

            pos_boxes = decoded_boxes[positive_indices]
            gt_iou = box_iou(first_gt, pos_boxes)
            for anchor_idx, (box, iou) in zip(positive_indices.tolist(), zip(pos_boxes.tolist(), gt_iou.tolist())):
                print(f"  Anchor {anchor_idx}: box={box}, IoU={iou:.4f}")

            displayed_indices = torch.tensor(selected_indices, dtype=torch.long, device=decoded_boxes.device)
            top_iou = box_iou(first_gt, decoded_boxes[displayed_indices])
            top_iou_list = top_iou.tolist()
            if isinstance(top_iou_list, float):
                top_iou_list = [top_iou_list]
            print("IoU of displayed anchors relative to first GT:")
            for anchor_idx, score, box, iou in zip(
                displayed_indices.tolist(),
                selected_scores,
                decoded_boxes[displayed_indices].tolist(),
                top_iou_list
            ):
                print(f"  idx={anchor_idx:4d} score={score:.4f} box={box} IoU={iou:.4f}")
        else:
            print("No positive anchors in this sample.")

        debug_path = create_debug_visualization(
            dataset=dataset,
            sample_idx=idx,
            decoded_boxes=decoded_boxes,
            top_indices=top_indices,
            top_scores=top_scores,
            anchor_targets=anchor_targets,
            class_predictions=class_predictions.squeeze(-1),
            output_root=Path("runs/logs"),
            comparison_detector=comparison_detector,
            comparison_label=args.compare_label
        )
        print(f"Saved debug visualization to {debug_path}")


if __name__ == "__main__":
    main()
