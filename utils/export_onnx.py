"""Export a BlazeFace checkpoint to ONNX for the browser demo.

The exported graph takes an RGB [1,3,128,128] tensor with pixels in [0,255] and
returns decoded boxes and sigmoid scores. Confidence filtering and NMS are left to
the caller, which keeps the graph free of data-dependent shapes.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from blazebase import anchor_options  # noqa: E402
from utils.model_utils import load_model  # noqa: E402


class WebBlazeFace(nn.Module):
    """Wraps the detector so the ONNX graph ends at decoded boxes + scores."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("anchors", model.anchors.clone())

    def forward(self, image_255):
        x = image_255 / 127.5 - 1.0
        raw, logits = self.model(x)
        ax, ay = self.anchors[:, 0], self.anchors[:, 1]
        aw, ah = self.anchors[:, 2], self.anchors[:, 3]
        xc = raw[..., 0] / 128.0 * aw + ax
        yc = raw[..., 1] / 128.0 * ah + ay
        w = raw[..., 2] / 128.0 * aw
        h = raw[..., 3] / 128.0 * ah
        boxes = torch.stack([yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2], dim=-1)
        return boxes, torch.sigmoid(logits).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cpu")
    model = load_model(str(REPO / args.weights) if not Path(args.weights).is_absolute()
                       else args.weights, dev)
    model.generate_anchors(anchor_options)
    net = WebBlazeFace(model).eval()

    dummy = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        b, s = net(dummy)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["image"], output_names=["boxes", "scores"],
        opset_version=12, dynamo=False,
    )
    print(f"{args.weights} -> {args.out}  boxes{tuple(b.shape)} scores{tuple(s.shape)} "
          f"({Path(args.out).stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
