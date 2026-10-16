"""Measure horizontal-flip test-time augmentation at matched precision.

Runs the detector on the image and its mirror, maps the mirrored boxes back, fuses
overlapping detections into a score-weighted mean, then suppresses overlaps again --
without that second suppression more views simply emit more boxes.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from utils.compare_detectors import iou_matrix  # noqa: E402
from utils.debug_training import _run_detector_on_image  # noqa: E402
from utils.model_utils import load_model  # noqa: E402


def fuse(boxes, scores, iou_threshold=0.55):
    """Cluster from the highest score down; each cluster becomes a weighted mean."""
    if len(boxes) == 0:
        return boxes, scores
    ious = iou_matrix(boxes, boxes)
    order = np.argsort(-scores)
    used = np.zeros(len(boxes), bool)
    fb, fs = [], []
    for i in order:
        if used[i]:
            continue
        members = np.where((ious[i] >= iou_threshold) & ~used)[0]
        used[members] = True
        w = scores[members][:, None]
        fb.append((boxes[members] * w).sum(0) / w.sum())
        fs.append(scores[members].max())
    return np.asarray(fb, np.float32), np.asarray(fs, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", default="data/splits/val.csv")
    ap.add_argument("--data-root", default="data/raw/blazeface")
    ap.add_argument("--sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--fuse-iou", type=float, default=0.55)
    args = ap.parse_args()

    root = Path(args.data_root)
    df = pd.read_csv(args.csv)
    groups = list(df.groupby("image_path", sort=False))
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(groups):
        groups = [groups[i] for i in
                  sorted(rng.choice(len(groups), size=args.sample, replace=False).tolist())]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(REPO / args.weights), dev, threshold=min(args.thresholds))

    print(f"{len(groups)} images | {args.weights}")
    print(f"  {'mode':>8} {'thr':>5} {'prec':>7} {'recall':>7}")
    for mode in ("single", "flip-tta"):
        for t in args.thresholds:
            model.min_score_thresh = t
            tp = fp = ngt = 0
            for rel, g in groups:
                im = cv2.imread(str(root / str(rel)))
                if im is None:
                    continue
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                xywh = g[["x1", "y1", "w", "h"]].dropna().to_numpy(np.float32)
                gt = (np.column_stack((xywh[:, 0], xywh[:, 1],
                                       xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]))
                      if len(xywh) else np.zeros((0, 4), np.float32))
                ngt += len(gt)
                b, s = _run_detector_on_image(model, rgb, dev)
                keep = s >= t
                b, s = b[keep], s[keep]
                if mode == "flip-tta":
                    fb, fs = _run_detector_on_image(
                        model, np.ascontiguousarray(rgb[:, ::-1]), dev)
                    k2 = fs >= t
                    fb, fs = fb[k2], fs[k2]
                    if len(fb):
                        W = im.shape[1]
                        fb = np.column_stack((W - fb[:, 2], fb[:, 1], W - fb[:, 0], fb[:, 3]))
                    b = np.concatenate([b, fb]) if len(fb) else b
                    s = np.concatenate([s, fs]) if len(fs) else s
                    b, s = fuse(b, s, args.fuse_iou)
                m = iou_matrix(b, gt)
                found = (m.max(axis=0) >= args.iou) if m.size else np.zeros(len(gt), bool)
                tp += int(found.sum())
                fp += len(b) - (int((m.max(axis=1) >= args.iou).sum()) if m.size else 0)
            print(f"  {mode:>8} {t:>5.2f} {tp/max(tp+fp,1):>7.4f} {tp/max(ngt,1):>7.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
