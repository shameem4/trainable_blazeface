"""Fit anchor priors to a dataset's face-size distribution.

The MediaPipe priors are 19/29 px (fine grid) and 45-111 px (coarse). If your faces
sit below that, the smallest anchor is a poor match and the band trains badly. Fitting
raises small-face recall at matched precision, at a small cost on larger faces.

Priors are used for ASSIGNMENT only -- inference still decodes with
fixed_anchor_size=True, so the exported graph and pretrained weights are unchanged.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def kmeans1d(x: np.ndarray, k: int, iters: int = 80) -> np.ndarray:
    c = np.quantile(x, np.linspace(0.5 / k, 1 - 0.5 / k, k))
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        nc = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return np.sort(nc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/raw/blazeface")
    ap.add_argument("--input-size", type=float, default=128.0)
    ap.add_argument("--min-px", type=float, default=8.0,
                    help="ignore faces below this; they are unresolvable at stride 8")
    ap.add_argument("--split-px", type=float, default=48.0,
                    help="fine grid (stride 8) owns faces below this, coarse above")
    ap.add_argument("--n-fine", type=int, default=2)
    ap.add_argument("--n-coarse", type=int, default=6)
    a = ap.parse_args()

    root = Path(a.data_root)
    df = pd.read_csv(a.csv)
    paths = df["image_path"].unique().tolist()

    def longest(p):
        with Image.open(root / str(p)) as im:
            return p, max(im.size)

    with ThreadPoolExecutor(max_workers=32) as ex:
        dims = dict(ex.map(longest, paths))

    px = (np.maximum(df["w"], df["h"]) / df["image_path"].map(dims) * a.input_size).to_numpy()
    px = px[np.isfinite(px)]
    fine = kmeans1d(px[(px >= a.min_px) & (px < a.split_px)], a.n_fine)
    coarse = kmeans1d(px[px >= a.split_px], a.n_coarse)

    np.save(a.out, np.concatenate([fine, coarse]) / a.input_size)
    print(f"fine priors   (px): {np.round(fine, 1).tolist()}")
    print(f"coarse priors (px): {np.round(coarse, 1).tolist()}")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
