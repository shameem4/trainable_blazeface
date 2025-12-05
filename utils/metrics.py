"""
Metrics and evaluation utilities (IoU, matching, etc.).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


def compute_iou(
    box1: tuple[int, int, int, int],
    box2: np.ndarray
) -> float:
    """Compute IoU between a ground truth box and a detection box."""
    gt_x1, gt_y1, gt_w, gt_h = box1
    gt_x2 = gt_x1 + gt_w
    gt_y2 = gt_y1 + gt_h

    det_y1, det_x1, det_y2, det_x2 = box2[0], box2[1], box2[2], box2[3]

    inter_x1 = max(gt_x1, det_x1)
    inter_y1 = max(gt_y1, det_y1)
    inter_x2 = min(gt_x2, det_x2)
    inter_y2 = min(gt_y2, det_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = gt_w * gt_h
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    union_area = gt_area + det_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def match_detections_to_ground_truth(
    gt_boxes: List[tuple[int, int, int, int]],
    detections: np.ndarray,
    iou_threshold: float = 0.3
) -> Tuple[List[int], List[float]]:
    """Match detections to ground truth boxes using IoU (greedy)."""
    if len(detections) == 0 or len(gt_boxes) == 0:
        return [], []

    num_gt = len(gt_boxes)
    num_det = len(detections)

    iou_matrix = np.zeros((num_gt, num_det))
    for i, gt_box in enumerate(gt_boxes):
        for j in range(num_det):
            iou_matrix[i, j] = compute_iou(gt_box, detections[j])

    matched_det_indices: List[int] = []
    matched_ious: List[float] = []
    used_detections = set()

    gt_order = np.argsort(-iou_matrix.max(axis=1))

    for gt_idx in gt_order:
        best_det_idx = -1
        best_iou = 0.0

        for det_idx in range(num_det):
            if det_idx in used_detections:
                continue
            iou = iou_matrix[gt_idx, det_idx]
            if iou > best_iou:
                best_iou = iou
                best_det_idx = det_idx

        if best_iou >= iou_threshold and best_det_idx != -1:
            matched_det_indices.append(best_det_idx)
            matched_ious.append(best_iou)
            used_detections.add(best_det_idx)
        else:
            matched_det_indices.append(-1)
            matched_ious.append(0.0)

    return matched_det_indices, matched_ious


def compute_iou_torch(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes ([ymin, xmin, ymax, xmax])."""
    y_min = torch.maximum(box1[:, None, 0], box2[None, :, 0])
    x_min = torch.maximum(box1[:, None, 1], box2[None, :, 1])
    y_max = torch.minimum(box1[:, None, 2], box2[None, :, 2])
    x_max = torch.minimum(box1[:, None, 3], box2[None, :, 3])

    intersection = torch.clamp(y_max - y_min, min=0) * torch.clamp(x_max - x_min, min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / (union + 1e-6)


def compute_mean_iou_torch(
    pred_boxes: torch.Tensor,
    true_boxes: torch.Tensor,
    scale: float = 128.0
) -> torch.Tensor:
    """Mean IoU between predicted and true boxes (MediaPipe convention)."""
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    pred_scaled = pred_boxes * scale
    true_scaled = true_boxes * scale

    y_min = torch.maximum(pred_scaled[:, 0], true_scaled[:, 0])
    x_min = torch.maximum(pred_scaled[:, 1], true_scaled[:, 1])
    y_max = torch.minimum(pred_scaled[:, 2], true_scaled[:, 2])
    x_max = torch.minimum(pred_scaled[:, 3], true_scaled[:, 3])

    intersection = torch.clamp(y_max - y_min + 1, min=0) * torch.clamp(x_max - x_min + 1, min=0)
    pred_area = (pred_scaled[:, 2] - pred_scaled[:, 0] + 1) * (pred_scaled[:, 3] - pred_scaled[:, 1] + 1)
    true_area = (true_scaled[:, 2] - true_scaled[:, 0] + 1) * (true_scaled[:, 3] - true_scaled[:, 1] + 1)
    union = pred_area + true_area - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean()


def compute_map_torch(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    true_boxes: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """Compute mAP for single-class detection."""
    if pred_boxes.numel() == 0 or true_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes[sorted_indices]

    num_gt = true_boxes.shape[0]
    gt_matched = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device)

    tp: List[int] = []
    fp: List[int] = []

    for pred_box in pred_boxes_sorted:
        ious = compute_iou_torch(pred_box.unsqueeze(0), true_boxes).squeeze(0)
        max_iou, max_idx = ious.max(dim=0)

        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[max_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    if not tp:
        return torch.tensor(0.0, device=pred_boxes.device)

    tp_tensor = torch.tensor(tp, dtype=torch.float32, device=pred_boxes.device)
    fp_tensor = torch.tensor(fp, dtype=torch.float32, device=pred_boxes.device)

    tp_cumsum = torch.cumsum(tp_tensor, dim=0)
    fp_cumsum = torch.cumsum(fp_tensor, dim=0)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)

    ap = torch.zeros(1, device=pred_boxes.device)
    for t in torch.linspace(0, 1, 11, device=pred_boxes.device):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max()
    ap /= 11
    return ap
