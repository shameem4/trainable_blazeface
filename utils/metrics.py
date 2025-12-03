"""
Metrics and evaluation utilities (IoU, matching, etc.).
"""
import numpy as np


def compute_iou(box1: tuple[int, int, int, int], box2: np.ndarray) -> float:
    """Compute IoU between a ground truth box and a detection box.

    Args:
        box1: Ground truth box in format (x1, y1, w, h)
        box2: Detection box in format [ymin, xmin, ymax, xmax, score]

    Returns:
        IoU value between 0 and 1
    """
    # Convert box1 from (x1, y1, w, h) to (x1, y1, x2, y2)
    gt_x1, gt_y1, gt_w, gt_h = box1
    gt_x2 = gt_x1 + gt_w
    gt_y2 = gt_y1 + gt_h

    # Detection box is already in (ymin, xmin, ymax, xmax) format
    det_y1, det_x1, det_y2, det_x2 = box2[0], box2[1], box2[2], box2[3]

    # Compute intersection
    inter_x1 = max(gt_x1, det_x1)
    inter_y1 = max(gt_y1, det_y1)
    inter_x2 = min(gt_x2, det_x2)
    inter_y2 = min(gt_y2, det_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Compute union
    gt_area = gt_w * gt_h
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    union_area = gt_area + det_area - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_detections_to_ground_truth(
    gt_boxes: list[tuple[int, int, int, int]],
    detections: np.ndarray,
    iou_threshold: float = 0.3
) -> tuple[list[int], list[float]]:
    """Match detections to ground truth boxes using IoU.

    Uses a greedy matching strategy: for each GT box, find the detection
    with highest IoU. Only keeps the best N detections where N = number of GT boxes.

    Args:
        gt_boxes: List of ground truth boxes in format (x1, y1, w, h)
        detections: Detection array [N, 5] with format [ymin, xmin, ymax, xmax, score]
        iou_threshold: Minimum IoU to consider a match

    Returns:
        - List of detection indices to keep (matched to GT boxes)
        - List of IoU values for each match
    """
    if len(detections) == 0 or len(gt_boxes) == 0:
        return [], []

    num_gt = len(gt_boxes)
    num_det = len(detections)

    # Compute IoU matrix: [num_gt x num_det]
    iou_matrix = np.zeros((num_gt, num_det))
    for i, gt_box in enumerate(gt_boxes):
        for j in range(num_det):
            iou_matrix[i, j] = compute_iou(gt_box, detections[j])

    # Greedy matching: for each GT box, find best detection
    matched_det_indices = []
    matched_ious = []
    used_detections = set()

    # Sort GT boxes by their maximum IoU (prioritize GT boxes with good matches)
    gt_order = np.argsort(-iou_matrix.max(axis=1))

    for gt_idx in gt_order:
        # Find best unused detection for this GT box
        best_det_idx = -1
        best_iou = 0.0

        for det_idx in range(num_det):
            if det_idx in used_detections:
                continue

            iou = iou_matrix[gt_idx, det_idx]
            if iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx

        # Only accept if above threshold
        if best_iou >= iou_threshold and best_det_idx != -1:
            matched_det_indices.append(best_det_idx)
            matched_ious.append(best_iou)
            used_detections.add(best_det_idx)
        else:
            # No good match for this GT box
            matched_det_indices.append(-1)
            matched_ious.append(0.0)

    return matched_det_indices, matched_ious
