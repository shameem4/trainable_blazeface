"""Regression tests for the v2 fixes (loss gradients, resume/epoch semantics, augmentation labels)."""
import unittest

import numpy as np
import torch

from loss_functions import BlazeFaceDetectionLoss
from utils import augmentation
from utils.metrics import compute_map_torch


class TestLogitLoss(unittest.TestCase):
    def test_saturated_logits_keep_gradient(self):
        # prob-based BCE clamps at 1e-7 and returns zero gradient at |logit| ~ 20
        for use_focal in (False, True):
            loss = BlazeFaceDetectionLoss(use_focal_loss=use_focal)
            bg = torch.tensor([20.0], requires_grad=True)
            loss.classification_loss_with_logits(bg, torch.zeros(1)).backward()
            self.assertGreater(bg.grad.abs().item(), 0.0)
            pos = torch.tensor([-20.0], requires_grad=True)
            loss.classification_loss_with_logits(pos, torch.ones(1)).backward()
            self.assertGreater(pos.grad.abs().item(), 0.0)

    def test_logit_and_prob_paths_agree(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 896, 1) * 2
        boxes = torch.randn(2, 896, 4)
        targets = torch.zeros(2, 896, 5)
        targets[:, 10:18, 0] = 1.0
        targets[:, 10:18, 1:] = torch.tensor([0.4, 0.4, 0.5, 0.5])
        anchors = torch.rand(896, 4)
        for use_focal in (False, True):
            fn = BlazeFaceDetectionLoss(use_focal_loss=use_focal)
            a = fn(torch.sigmoid(logits), boxes, targets, anchors)
            b = fn(torch.sigmoid(logits), boxes, targets, anchors, class_logits=logits)
            for k in ("background", "positive", "detection"):
                self.assertAlmostEqual(a[k].item(), b[k].item(), places=4)


class TestAugmentationKeepsTargetsVisible(unittest.TestCase):
    def test_occlusion_and_cutout_never_hide_box(self):
        np.random.seed(0)
        box = np.array([[0.30, 0.30, 0.70, 0.70]], dtype=np.float32)
        for fn, kw in ((augmentation.augment_synthetic_occlusion, {"occlusion_size_range": (30, 60)}),
                       (augmentation.augment_cutout, {"hole_size_range": (30, 60)})):
            for _ in range(200):
                img = np.full((128, 128, 3), 255, dtype=np.uint8)
                out = fn(img.copy(), bboxes=box, **kw)
                region = out[int(0.3 * 128):int(0.7 * 128), int(0.3 * 128):int(0.7 * 128)]
                hidden = float((region != 255).any(axis=-1).mean())
                self.assertLessEqual(hidden, 0.26)

    def test_rotation_does_not_inflate_more_than_hull(self):
        np.random.seed(1)
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        box = np.array([[0.4, 0.4, 0.6, 0.6]], dtype=np.float32)
        _, out = augmentation.augment_rotation(img, box.copy(), angle_range=(10, 10))
        side = out[0, 2] - out[0, 0]
        hull = 0.2 * (np.cos(np.deg2rad(10)) + np.sin(np.deg2rad(10)))
        self.assertLess(side, hull)
        self.assertGreaterEqual(side, 0.2 - 1e-6)


class TestMetric(unittest.TestCase):
    def test_perfect_detection_ap_is_one(self):
        gt = torch.tensor([[0.1, 0.1, 0.3, 0.3]])
        ap = compute_map_torch(gt.clone(), torch.tensor([0.9]), gt)
        self.assertAlmostEqual(ap.item(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()


class TestNegativeImages(unittest.TestCase):
    def test_empty_box_rows_become_negative_samples(self):
        import tempfile
        from pathlib import Path
        from dataloader import CSVDetectorDataset, collate_detector_fn
        from utils.data_utils import load_image_boxes_from_csv
        src = Path(__file__).resolve().parents[1] / "assets"
        with tempfile.TemporaryDirectory() as d:
            csv = Path(d) / "t.csv"
            csv.write_text("image_path,x1,y1,w,h\nimages/single_face.jpg,10,10,40,40\nimages/multi_face.jpg,,,,\n")
            ds = CSVDetectorDataset(str(csv), str(src), augment=True)
            self.assertEqual([len(s["boxes"]) for s in ds.samples], [1, 0])
            batch = collate_detector_fn([ds[0], ds[1]])
            self.assertEqual(float(batch["anchor_targets"][1, :, 0].sum()), 0.0)
            self.assertEqual(batch["gt_box_counts"].tolist(), [1, 0])
            paths, boxes = load_image_boxes_from_csv(csv)
            self.assertEqual(boxes["images/multi_face.jpg"], [])
