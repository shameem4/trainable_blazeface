import argparse
from pathlib import Path

import torch

from csv_dataloader import CSVDetectorDataset
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug BlazeFace training sample (single image end-to-end)"
    )
    parser.add_argument("--csv", type=str, default="data/splits/train_new.csv")
    parser.add_argument("--data-root", type=str, default="data/raw/blazeface")
    parser.add_argument("--weights", type=str, default="model_weights/blazeface.pth")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect")
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
    else:
        print("No positive anchors in this sample.")


if __name__ == "__main__":
    main()
