"""Drop ground-truth boxes too small for the 128x128 model to resolve.

The finest anchor stride is 8 px at the model's input scale, so faces below roughly
16 px there are not learnable. Training on them costs precision on the faces that are
(see the Experimental Results section of the README).

Images that would lose more than `--max-dropped-frac` of their faces are removed
entirely rather than kept: letting real faces become background labels teaches the
detector to suppress exactly what it should find.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/raw/blazeface")
    ap.add_argument("--min-px", type=float, default=16.0,
                    help="minimum face size at the 128px input scale")
    ap.add_argument("--input-size", type=float, default=128.0)
    ap.add_argument("--max-dropped-frac", type=float, default=0.5,
                    help="drop an image if more than this fraction of its faces are too small")
    args = ap.parse_args()

    root = Path(args.data_root)
    df = pd.read_csv(args.csv)
    paths = df["image_path"].unique().tolist()

    def longest_side(p):
        with Image.open(root / str(p)) as im:
            return p, max(im.size)

    with ThreadPoolExecutor(max_workers=32) as ex:
        dims = dict(ex.map(longest_side, paths))

    # Images are letterboxed to a square, so the longest side sets the scale.
    px = np.maximum(df["w"], df["h"]) / df["image_path"].map(dims) * args.input_size
    keep = px >= args.min_px
    frac_kept = keep.groupby(df["image_path"]).mean()
    good = set(frac_kept[frac_kept > (1.0 - args.max_dropped_frac)].index)

    out = df[keep & df["image_path"].isin(good)][["image_path", "x1", "y1", "w", "h"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n_img = out["image_path"].nunique()
    print(f"{args.csv} -> {args.out}")
    print(f"  images {n_img}/{len(paths)} ({n_img / len(paths) * 100:.1f}%)  "
          f"boxes {len(out)}/{len(df)} ({len(out) / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
