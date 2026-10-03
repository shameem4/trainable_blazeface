import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

# Import from consolidated utils modules
from utils.anchor_utils import (
    generate_reference_anchors,
    encode_boxes_to_anchors,
    flatten_anchor_targets,
    anchor_options,
)
from utils.iou import compute_iou_np as compute_iou


def load_mediapipe_weights(model: nn.Module,
                           weights_path: str,
                           strict: bool = False,
                           load_detection_heads: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load MediaPipe pretrained weights into the (BatchNorm-free, BN-folded) BlazeFace model.

    Args:
        model: BlazeFace detector model
        weights_path: Path to .pth file with MediaPipe weights
        strict: Whether to require exact match
        load_detection_heads: Whether to load classifier/regressor heads (default: True).
                             Set to False when fine-tuning for a different detection task.

    Returns:
        missing_keys: Keys in model not found in weights
        unexpected_keys: Keys in weights not found in model
    """
    def maybe_strip(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if load_detection_heads:
            return state_dict
        filtered = {
            k: v for k, v in state_dict.items()
            if 'classifier' not in k and 'regressor' not in k
        }
        return filtered
    
    def split_regressor_heads(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Handle legacy single-regressor heads by splitting box/kp channels."""
        mapping = [
            ('regressor_8', 'regressor_8_box', 'regressor_8_kp', 2),
            ('regressor_16', 'regressor_16_box', 'regressor_16_kp', 6),
        ]
        updated = dict(state_dict)
        total_coords = 16
        box_coords = 4
        kp_coords = total_coords - box_coords
        for base, box_key, kp_key, anchors_per_cell in mapping:
            weight_key = f"{base}.weight"
            bias_key = f"{base}.bias"
            if weight_key in updated:
                weight = updated.pop(weight_key)
                out_channels, in_channels, kh, kw = weight.shape
                weight = weight.view(anchors_per_cell, total_coords, in_channels, kh, kw)
                box_weight = weight[:, :box_coords].reshape(-1, in_channels, kh, kw)
                kp_weight = weight[:, box_coords:].reshape(-1, in_channels, kh, kw)
                updated[f"{box_key}.weight"] = box_weight.clone()
                updated[f"{kp_key}.weight"] = kp_weight.clone()
            if bias_key in updated:
                bias = updated.pop(bias_key)
                bias = bias.view(anchors_per_cell, total_coords)
                box_bias = bias[:, :box_coords].reshape(-1)
                kp_bias = bias[:, box_coords:].reshape(-1)
                updated[f"{box_key}.bias"] = box_bias.clone()
                updated[f"{kp_key}.bias"] = kp_bias.clone()
        return updated

    original_state = torch.load(weights_path, map_location='cpu')
    state_dict = split_regressor_heads(maybe_strip(original_state))
    result = model.load_state_dict(state_dict, strict=strict)
    return result.missing_keys, result.unexpected_keys


# =============================================================================
# Model Building Blocks
# =============================================================================

class BlazeBlock_WT(nn.Module):
    """
    BlazeBlock with BatchNorm folded into the conv weights/biases (no explicit BN layers),
    matching MediaPipe's converted .pth weights. This is the block the model actually uses and
    trains; the folded conv weights/biases are fine-tuned directly.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='relu', skip_proj=False):
        super(BlazeBlock_WT, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        if skip_proj:
            self.skip_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.skip_proj = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s"%act)

    def forward(self, x):
        if self.stride == 2:
            if self.kernel_size==3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.skip_proj is not None:
            x = self.skip_proj(x)
        elif self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(FinalBlazeBlock, self).__init__()

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, stride=2, padding=0,
                      groups=channels, bias=True),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)

        return self.act(self.convs(h))


class BlazeBase(nn.Module):
    """ Base class for media pipe models. """
    
    # Type annotations for class attributes
    detection2roi_method: str
    kp1: int
    kp2: int
    dy: float
    dscale: float
    theta0: float
    
    # Training-related attributes
    reference_anchors: Optional[torch.Tensor] = None
    small_anchors: Optional[torch.Tensor] = None
    big_anchors: Optional[torch.Tensor] = None


    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return next(self.parameters()).device
    
    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
        if hasattr(self, "generate_anchors"):
            self.generate_anchors(anchor_options)  # type: ignore[attr-defined]
    
    def init_anchors(self, input_size: int = 128) -> None:
        """
        Initialize reference anchors for training/inference.
        
        Args:
            input_size: Input image size (128 for front model)
        """
        self.reference_anchors, self.small_anchors, self.big_anchors = \
            generate_reference_anchors(input_size)
        
        # Move to device
        device = self._device()
        self.reference_anchors = self.reference_anchors.to(device)
        self.small_anchors = self.small_anchors.to(device)
        self.big_anchors = self.big_anchors.to(device)

    def detection2roi(
        self,
        detection: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """


        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        return xc, yc, scale, theta

