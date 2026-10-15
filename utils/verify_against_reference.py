"""Check this repo's BlazeFace port against Google's official MediaPipe detector.

Runs both on identical images and ground truth. They should agree closely; if they
do not, a measured gap between stock and a fine-tuned checkpoint may be a porting
artifact rather than a real result.

Needs the official model, which is not vendored:

    curl -L -o blaze_face_short_range.tflite \\
      https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

and the `mediapipe` package (`pip install mediapipe`).
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

BINS = [(0, 4), (4, 8), (8, 16), (16, 32), (32, float("inf"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-model", required=True,
                    help="path to blaze_face_short_range.tflite")
    ap.add_argument("--csv", default="data/splits/val.csv")
    ap.add_argument("--data-root", default="data/raw/blazeface")
    ap.add_argument("--weights", default="model_weights/blazeface.pth")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--input-size", type=float, default=128.0)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    root = Path(args.data_root)
    df = pd.read_csv(args.csv)
    groups = list(df.groupby("image_path", sort=False))
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(groups):
        idx = sorted(rng.choice(len(groups), size=args.sample, replace=False).tolist())
        groups = [groups[i] for i in idx]

    reference = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.reference_model),
        min_detection_confidence=args.threshold))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours = load_model(str(REPO / args.weights), device, threshold=args.threshold)

    acc = {k: dict(hit=np.zeros(len(BINS)), tot=np.zeros(len(BINS)), tp=0, fp=0)
           for k in ("google_reference", "this_repo_stock")}

    for rel, grp in groups:
        bgr = cv2.imread(str(root / str(rel)))
        if bgr is None:
            continue
        xywh = grp[["x1", "y1", "w", "h"]].dropna().to_numpy(np.float32)
        if not len(xywh):
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = np.column_stack((xywh[:, 0], xywh[:, 1],
                              xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]))
        sizes = np.maximum(xywh[:, 2], xywh[:, 3]) / max(bgr.shape[:2]) * args.input_size

        res = reference.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        ref_boxes = np.array([[d.bounding_box.origin_x, d.bounding_box.origin_y,
                               d.bounding_box.origin_x + d.bounding_box.width,
                               d.bounding_box.origin_y + d.bounding_box.height]
                              for d in res.detections], np.float32).reshape(-1, 4)
        our_boxes, our_scores = _run_detector_on_image(ours, rgb, device)
        our_boxes = our_boxes[our_scores >= args.threshold]

        for name, boxes in (("google_reference", ref_boxes), ("this_repo_stock", our_boxes)):
            a = acc[name]
            m = iou_matrix(boxes, gt)
            found = (m.max(axis=0) >= args.iou) if m.size else np.zeros(len(gt), bool)
            a["tp"] += int(found.sum())
            a["fp"] += len(boxes) - (int((m.max(axis=1) >= args.iou).sum()) if m.size else 0)
            for bi, (lo, hi) in enumerate(BINS):
                sel = (sizes >= lo) & (sizes < hi)
                a["tot"][bi] += sel.sum()
                a["hit"][bi] += found[sel].sum()

    for name, a in acc.items():
        bins_s = "  ".join(
            f"{lo}-{'inf' if hi == float('inf') else int(hi)}px "
            f"{(a['hit'][bi] / a['tot'][bi]):.3f}" if a["tot"][bi] else ""
            for bi, (lo, hi) in enumerate(BINS))
        print(f"{name:>18}  recall {a['hit'].sum() / a['tot'].sum():.4f}  "
              f"precision {a['tp'] / max(a['tp'] + a['fp'], 1):.4f}")
        print(f"{'':>18}  {bins_s}")


if __name__ == "__main__":
    main()
