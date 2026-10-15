"""Compare face detectors on the val split: PR curve at IoU>=0.5, plus mirror consistency.

BlazeFace's weighted NMS blends every detection above min_score_thresh, so its
output depends on the threshold. Each operating point therefore gets its own
forward pass at that threshold; a single low-floor pass cannot be swept.
"""
import argparse
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.debug_training import _run_detector_on_image  # noqa: E402
from utils.model_utils import load_model  # noqa: E402


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred [N,4] xyxy, gt [M,4] xyxy -> [N,M]."""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    x1 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y1 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x2 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y2 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = area_p[:, None] + area_g[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float):
    """Greedy match in descending score order. Returns is_tp aligned to score order."""
    order = np.argsort(-scores)
    pred, scores = pred[order], scores[order]
    ious = iou_matrix(pred, gt)
    taken = np.zeros(len(gt), dtype=bool)
    is_tp = np.zeros(len(pred), dtype=bool)
    for i in range(len(pred)):
        if len(gt) == 0:
            break
        cand = np.where(~taken)[0]
        if cand.size == 0:
            break
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= thr:
            is_tp[i] = True
            taken[j] = True
    return scores, is_tp


def flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    out = boxes.copy()
    out[:, 0] = width - boxes[:, 2]
    out[:, 2] = width - boxes[:, 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/splits/val.csv")
    ap.add_argument("--data-root", default="data/raw/blazeface")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="label=path pairs")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.3, 0.5, 0.75, 0.9],
                    help="operating points; detector is re-run at each")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="random subset of this size (val.csv is ordered by "
                         "category, so a head slice is not representative)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)

    df = pd.read_csv(args.csv)
    groups = list(df.groupby("image_path", sort=False))
    if args.max_images:
        groups = groups[: args.max_images]
    if args.sample and args.sample < len(groups):
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(len(groups), size=args.sample, replace=False).tolist())
        groups = [groups[i] for i in idx]
    print(f"{len(groups)} images from {args.csv}")

    def load(rel):
        img = cv2.imread(str(root / str(rel)))
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rels = [g[0] for g in groups]

    def prefetch(items, depth=24):
        """Decode ahead by at most `depth` images, so RAM stays bounded."""
        with ThreadPoolExecutor(max_workers=12) as ex:
            pending = deque()
            it = iter(items)
            for rel in islice(it, depth):
                pending.append(ex.submit(load, rel))
            for rel in it:
                yield pending.popleft().result()
                pending.append(ex.submit(load, rel))
            while pending:
                yield pending.popleft().result()

    gts = []
    for _, g in groups:
        xywh = g[["x1", "y1", "w", "h"]].dropna().to_numpy(dtype=np.float32)
        if xywh.size:
            gts.append(np.column_stack((xywh[:, 0], xywh[:, 1],
                                        xywh[:, 0] + xywh[:, 2],
                                        xywh[:, 1] + xywh[:, 3])))
        else:
            gts.append(np.zeros((0, 4), dtype=np.float32))

    total_gt = sum(len(g) for g in gts)
    print(f"{total_gt} ground-truth boxes\n")

    for spec in args.weights:
        label, path = spec.split("=", 1)
        print(f"--- {label} ({path}) ---")
        model = load_model(path, device, threshold=args.thresholds[0])

        # BlazeFace uses WEIGHTED NMS: it blends the coords and scores of every
        # detection above min_score_thresh. The output therefore depends on the
        # threshold itself, so one low-floor pass cannot be swept afterwards --
        # each operating point needs its own forward pass at that threshold.
        print(f"  {'thresh':>7} {'prec':>7} {'recall':>7} {'TP':>6} {'FP':>6} {'mirror':>7}")
        for t in args.thresholds:
            model.min_score_thresh = t
            tp_n = fp_n = 0
            agree_num = agree_den = 0
            # Stream images: decoding 8k full-size frames at once is tens of GB.
            for img, gt in zip(prefetch(rels), gts):
                if img is None:
                    continue
                boxes, scores = _run_detector_on_image(model, img, device)
                keep = scores >= t
                boxes, scores = boxes[keep], scores[keep]
                _, is_tp = match(boxes, scores, gt, args.iou)
                tp_n += int(is_tp.sum())
                fp_n += int((~is_tp).sum())

                if args.mirror:
                    fb, fs = _run_detector_on_image(
                        model, np.ascontiguousarray(img[:, ::-1]), device)
                    fb = flip_boxes(fb[fs >= t], img.shape[1])
                    if len(boxes) or len(fb):
                        m = iou_matrix(boxes, fb)
                        agree_num += 0 if m.size == 0 else int((m.max(axis=1) >= 0.5).sum())
                        agree_den += max(len(boxes), len(fb))

            prec = tp_n / max(tp_n + fp_n, 1)
            rec = tp_n / max(total_gt, 1)
            mir = f"{agree_num / agree_den:.4f}" if agree_den else "-"
            print(f"  {t:>7.2f} {prec:>7.4f} {rec:>7.4f} {tp_n:>6} {fp_n:>6} {mir:>7}",
                  flush=True)

        if args.mirror and agree_den:
            print(f"  mirror agreement @0.5: {agree_num / agree_den:.4f} "
                  f"({agree_num}/{agree_den})")
        print()


if __name__ == "__main__":
    main()
