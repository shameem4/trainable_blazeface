import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


# =============================================================================
# Anchor Generation Utilities (following vincent1bt/blazeface-tensorflow)
# =============================================================================

def generate_reference_anchors(
    input_size: int = 128,
    fixed_anchor_size: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate reference anchor centers for BlazeFace detector.
    
    Creates a grid of anchor centers for two scales:
    - 16x16 grid with 2 anchors per cell = 512 small anchors
    - 8x8 grid with 6 anchors per cell = 384 big anchors
    Total: 896 anchors
    
    Args:
        input_size: Input image size (default 128)
        fixed_anchor_size: If True, all anchors have w=h=1.0 (default).
                          If False, could support variable anchor sizes in future.
        
    Returns:
        reference_anchors: [896, 4] tensor of (x_center, y_center, width, height)
        small_anchors: [512, 4] tensor for 16x16 grid
        big_anchors: [384, 4] tensor for 8x8 grid
    """
    # Small anchors: 16x16 grid, size 0.0625 (1/16)
    # Centers at 0.03125, 0.09375, ..., 0.96875
    small_boxes = torch.linspace(0.03125, 0.96875, 16)
    
    # Big anchors: 8x8 grid, size 0.125 (1/8)  
    # Centers at 0.0625, 0.1875, ..., 0.9375
    big_boxes = torch.linspace(0.0625, 0.9375, 8)
    
    # Create grid for small anchors (16x16 with 2 anchors per cell = 512)
    # x coordinates: repeat each x 2 times, then tile 16 times
    small_x = small_boxes.repeat_interleave(2).repeat(16)  # 512
    # y coordinates: repeat each y 32 times (2 anchors * 16 x positions)
    small_y = small_boxes.repeat_interleave(32)  # 512
    # Width and height: 1.0 for fixed_anchor_size=True
    small_w = torch.ones_like(small_x)
    small_h = torch.ones_like(small_x)
    small_anchors = torch.stack([small_x, small_y, small_w, small_h], dim=1)  # [512, 4]
    
    # Create grid for big anchors (8x8 with 6 anchors per cell = 384)
    # x coordinates: repeat each x 6 times, then tile 8 times
    big_x = big_boxes.repeat_interleave(6).repeat(8)  # 384
    # y coordinates: repeat each y 48 times (6 anchors * 8 x positions)
    big_y = big_boxes.repeat_interleave(48)  # 384
    # Width and height: 1.0 for fixed_anchor_size=True
    big_w = torch.ones_like(big_x)
    big_h = torch.ones_like(big_x)
    big_anchors = torch.stack([big_x, big_y, big_w, big_h], dim=1)  # [384, 4]
    
    # Combine: small first, then big (matching model output order)
    reference_anchors = torch.cat([small_anchors, big_anchors], dim=0)  # [896, 4]
    
    return reference_anchors, small_anchors, big_anchors


def compute_iou(box: np.ndarray, anchor_box: np.ndarray) -> float:
    """
    Compute IoU between a ground truth box and an anchor box.
    
    Args:
        box: [x1, y1, x2, y2] ground truth box (in pixel coordinates)
        anchor_box: [x1, y1, x2, y2] anchor box (in pixel coordinates)
        
    Returns:
        IoU value
    """
    x_min = max(box[0], anchor_box[0])
    y_min = max(box[1], anchor_box[1])
    x_max = min(box[2], anchor_box[2])
    y_max = min(box[3], anchor_box[3])
    
    overlap_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    anchor_area = (anchor_box[2] - anchor_box[0]) * (anchor_box[3] - anchor_box[1])
    
    union_area = box_area + anchor_area - overlap_area
    
    if union_area == 0:
        return 0.0
    
    return overlap_area / union_area


def encode_boxes_to_anchors(
    boxes: np.ndarray,
    input_size: int = 128,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match ground truth boxes to anchors and encode targets.
    
    Following vincent1bt's approach:
    - Match each GT box to anchors with best IoU
    - Encode targets as [class, x1, y1, x2, y2] normalized
    
    Args:
        boxes: [N, 4] array of boxes in [x, y, w, h] format (normalized 0-1)
        input_size: Input image size
        iou_threshold: Minimum IoU for anchor matching
        
    Returns:
        small_targets: [16, 16, 5] targets for 16x16 grid
        big_targets: [8, 8, 5] targets for 8x8 grid
    """
    # Anchor grid definitions
    small_grid = np.linspace(0.03125, 0.96875, 16).astype(np.float32)
    big_grid = np.linspace(0.0625, 0.9375, 8).astype(np.float32)
    
    small_size = 0.0625  # 1/16
    big_size = 0.125     # 1/8
    
    # Initialize target arrays: [y, x, 5] where 5 = [class, x1, y1, x2, y2]
    small_targets = np.zeros((16, 16, 5), dtype=np.float32)
    big_targets = np.zeros((8, 8, 5), dtype=np.float32)
    
    for box in boxes:
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        x, y, w, h = box
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        face_box = np.array([x1, y1, x2, y2])
        
        # Scale to pixel coordinates for IoU computation
        face_box_pixels = face_box * input_size
        
        # Try to match to small anchors (16x16 grid)
        best_small_iou = 0
        best_small_idx = None
        
        for y_idx, y_coord in enumerate(small_grid):
            for x_idx, x_coord in enumerate(small_grid):
                # Anchor box corners
                ax1 = x_coord - small_size / 2
                ay1 = y_coord - small_size / 2
                ax2 = x_coord + small_size / 2
                ay2 = y_coord + small_size / 2
                anchor_box = np.array([ax1, ay1, ax2, ay2]) * input_size
                
                iou = compute_iou(face_box_pixels, anchor_box)
                if iou > best_small_iou:
                    best_small_iou = iou
                    best_small_idx = (y_idx, x_idx)
        
        # Try to match to big anchors (8x8 grid)
        best_big_iou = 0
        best_big_idx = None
        
        for y_idx, y_coord in enumerate(big_grid):
            for x_idx, x_coord in enumerate(big_grid):
                # Anchor box corners
                ax1 = x_coord - big_size / 2
                ay1 = y_coord - big_size / 2
                ax2 = x_coord + big_size / 2
                ay2 = y_coord + big_size / 2
                anchor_box = np.array([ax1, ay1, ax2, ay2]) * input_size
                
                iou = compute_iou(face_box_pixels, anchor_box)
                if iou > best_big_iou:
                    best_big_iou = iou
                    best_big_idx = (y_idx, x_idx)
        
        # Assign to best matching anchor(s) if IoU exceeds threshold
        # Or assign to best anchor regardless (for training stability)
        if best_small_idx is not None and best_small_iou > 0:
            y_idx, x_idx = best_small_idx
            small_targets[y_idx, x_idx] = [1.0, x1, y1, x2, y2]
        
        if best_big_idx is not None and best_big_iou > 0:
            y_idx, x_idx = best_big_idx
            big_targets[y_idx, x_idx] = [1.0, x1, y1, x2, y2]
    
    return small_targets, big_targets


def flatten_anchor_targets(
    small_targets: np.ndarray,
    big_targets: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten anchor targets to match model output shape.
    
    Args:
        small_targets: [16, 16, 5] from 16x16 grid
        big_targets: [8, 8, 5] from 8x8 grid
        
    Returns:
        classes: [896] tensor of class labels (0 or 1)
        coords: [896, 4] tensor of box coordinates [x1, y1, x2, y2]
    """
    # Reshape to [num_anchors, 5]
    # Small: 16x16 grid with 2 anchors per cell
    # We need to expand to match the 512 anchors
    small_flat = small_targets.reshape(-1, 5)  # [256, 5]
    # Repeat each anchor 2 times (2 anchors per cell)
    small_flat = np.repeat(small_flat, 2, axis=0)  # [512, 5]
    
    # Big: 8x8 grid with 6 anchors per cell
    big_flat = big_targets.reshape(-1, 5)  # [64, 5]
    # Repeat each anchor 6 times (6 anchors per cell)
    big_flat = np.repeat(big_flat, 6, axis=0)  # [384, 5]
    
    # Combine
    all_targets = np.concatenate([small_flat, big_flat], axis=0)  # [896, 5]
    
    classes = torch.from_numpy(all_targets[:, 0])  # [896]
    coords = torch.from_numpy(all_targets[:, 1:])  # [896, 4]
    
    return classes, coords


# =============================================================================
# Weight Conversion: BlazeBlock_WT -> BlazeBlock
# =============================================================================

def unfold_conv_bn(conv_weight: torch.Tensor, conv_bias: torch.Tensor, 
                   num_features: int, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    "Unfold" a convolution with folded BatchNorm back to separate conv + BN.
    
    When BatchNorm is folded into conv, the transformation is:
        W_folded = W * gamma / sqrt(var + eps)
        b_folded = (b - mean) * gamma / sqrt(var + eps) + beta
    
    To reverse this, we assume the original BN had:
        gamma = 1 (scale)
        beta = 0 (shift)  
        mean = 0
        var = 1
    
    This gives us a valid starting point for fine-tuning where:
        - Conv weights are copied as-is
        - BN is initialized to identity transform
        - The bias is moved to BN beta
    
    Args:
        conv_weight: [out_ch, in_ch, kH, kW] conv weight with folded BN
        conv_bias: [out_ch] conv bias with folded BN
        num_features: number of features for BatchNorm
        eps: BatchNorm epsilon
        
    Returns:
        new_conv_weight: [out_ch, in_ch, kH, kW] 
        bn_weight (gamma): [num_features] initialized to 1
        bn_bias (beta): [num_features] initialized to conv_bias
        bn_running_mean: [num_features] initialized to 0
        bn_running_var: [num_features] initialized to 1
    """
    # Conv weight stays the same
    new_conv_weight = conv_weight.clone()
    
    # BatchNorm parameters: initialize to identity transform
    # gamma (weight) = 1, so output = normalized * 1 = normalized
    bn_weight = torch.ones(num_features)
    
    # beta (bias) absorbs the original conv bias
    # This way: conv(x) + bias ≈ BN(conv(x)) when BN is identity + bias
    bn_bias = conv_bias.clone()
    
    # Running statistics: mean=0, var=1 (identity normalization initially)
    bn_running_mean = torch.zeros(num_features)
    bn_running_var = torch.ones(num_features)
    
    return new_conv_weight, bn_weight, bn_bias, bn_running_mean, bn_running_var


def convert_blazeblock_weights(old_state: Dict[str, torch.Tensor], 
                                prefix: str = ""
) -> Dict[str, torch.Tensor]:
    """
    Convert a single BlazeBlock_WT's weights to BlazeBlock format.
    
    Maps:
        convs.0.weight, convs.0.bias -> dw_conv.weight, bn1.*
        convs.1.weight, convs.1.bias -> pw_conv.weight, bn2.*
        skip_proj.weight, skip_proj.bias -> skip_proj.0.weight, skip_proj.1.*
    
    Args:
        old_state: State dict containing BlazeBlock_WT weights
        prefix: Key prefix (e.g., "backbone1.2.")
        
    Returns:
        New state dict with BlazeBlock weight names
    """
    new_state = {}
    
    # Depthwise conv: convs.0 -> dw_conv + bn1
    dw_weight = old_state[f"{prefix}convs.0.weight"]
    dw_bias = old_state[f"{prefix}convs.0.bias"]
    num_dw_features = dw_weight.shape[0]
    
    new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
        dw_weight, dw_bias, num_dw_features
    )
    
    new_state[f"{prefix}dw_conv.weight"] = new_weight
    new_state[f"{prefix}bn1.weight"] = bn_w
    new_state[f"{prefix}bn1.bias"] = bn_b
    new_state[f"{prefix}bn1.running_mean"] = bn_mean
    new_state[f"{prefix}bn1.running_var"] = bn_var
    new_state[f"{prefix}bn1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # Pointwise conv: convs.1 -> pw_conv + bn2
    pw_weight = old_state[f"{prefix}convs.1.weight"]
    pw_bias = old_state[f"{prefix}convs.1.bias"]
    num_pw_features = pw_weight.shape[0]
    
    new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
        pw_weight, pw_bias, num_pw_features
    )
    
    new_state[f"{prefix}pw_conv.weight"] = new_weight
    new_state[f"{prefix}bn2.weight"] = bn_w
    new_state[f"{prefix}bn2.bias"] = bn_b
    new_state[f"{prefix}bn2.running_mean"] = bn_mean
    new_state[f"{prefix}bn2.running_var"] = bn_var
    new_state[f"{prefix}bn2.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # Skip projection (if exists): skip_proj -> skip_proj.0 + skip_proj.1
    skip_weight_key = f"{prefix}skip_proj.weight"
    if skip_weight_key in old_state:
        skip_weight = old_state[skip_weight_key]
        skip_bias = old_state[f"{prefix}skip_proj.bias"]
        num_skip_features = skip_weight.shape[0]
        
        new_weight, bn_w, bn_b, bn_mean, bn_var = unfold_conv_bn(
            skip_weight, skip_bias, num_skip_features
        )
        
        new_state[f"{prefix}skip_proj.0.weight"] = new_weight
        new_state[f"{prefix}skip_proj.1.weight"] = bn_w
        new_state[f"{prefix}skip_proj.1.bias"] = bn_b
        new_state[f"{prefix}skip_proj.1.running_mean"] = bn_mean
        new_state[f"{prefix}skip_proj.1.running_var"] = bn_var
        new_state[f"{prefix}skip_proj.1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    
    # PReLU weight (if exists)
    prelu_key = f"{prefix}act.weight"
    if prelu_key in old_state:
        new_state[prelu_key] = old_state[prelu_key].clone()
    
    return new_state


def convert_blazeface_wt_to_trainable(old_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert full BlazeFace state dict from BlazeBlock_WT to BlazeBlock format.
    
    This enables loading pretrained MediaPipe weights into a trainable
    BlazeBlock-based model with explicit BatchNorm layers.
    
    Note: Regressor weights are converted to extract only box coordinates (first 4 of 16),
    discarding keypoint regression channels since our trainable model is box-only.
    
    Args:
        old_state_dict: State dict from model using BlazeBlock_WT
        
    Returns:
        New state dict compatible with trainable BlazeBlock model
    """
    new_state_dict = {}
    processed_prefixes = set()
    
    for key in old_state_dict.keys():
        # Check if this is a BlazeBlock_WT conv layer
        if ".convs.0.weight" in key:
            # Extract prefix (e.g., "backbone1.2.")
            prefix = key.replace("convs.0.weight", "")
            
            if prefix not in processed_prefixes:
                processed_prefixes.add(prefix)
                # Convert this block
                block_state = convert_blazeblock_weights(old_state_dict, prefix)
                new_state_dict.update(block_state)
        
        # Copy non-BlazeBlock_WT weights directly (classifiers, regressors, etc.)
        elif ".convs." not in key and ".skip_proj." not in key:
            # Check if it's not already processed as part of a block
            is_block_key = any(key.startswith(p) for p in processed_prefixes)
            if not is_block_key or "act.weight" not in key:
                # Handle regressor weights - extract box-only channels
                if "regressor" in key:
                    weight = old_state_dict[key]
                    if "regressor_8" in key:
                        # regressor_8: 32 -> 8 channels (2 anchors × 16 coords -> 2 anchors × 4 coords)
                        # Extract channels [0:4, 16:20] for 2 anchors' box coords
                        if weight.dim() == 4:  # conv weight
                            new_weight = torch.cat([weight[0:4], weight[16:20]], dim=0)
                        else:  # bias
                            new_weight = torch.cat([weight[0:4], weight[16:20]], dim=0)
                        new_state_dict[key] = new_weight
                    elif "regressor_16" in key:
                        # regressor_16: 96 -> 24 channels (6 anchors × 16 coords -> 6 anchors × 4 coords)
                        # Extract channels [0:4, 16:20, 32:36, 48:52, 64:68, 80:84] for 6 anchors' box coords
                        if weight.dim() == 4:  # conv weight
                            new_weight = torch.cat([
                                weight[0:4], weight[16:20], weight[32:36],
                                weight[48:52], weight[64:68], weight[80:84]
                            ], dim=0)
                        else:  # bias
                            new_weight = torch.cat([
                                weight[0:4], weight[16:20], weight[32:36],
                                weight[48:52], weight[64:68], weight[80:84]
                            ], dim=0)
                        new_state_dict[key] = new_weight
                else:
                    new_state_dict[key] = old_state_dict[key].clone()
    
    return new_state_dict


def load_mediapipe_weights(model: nn.Module,
                           weights_path: str,
                           strict: bool = False,
                           load_detection_heads: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Load MediaPipe pretrained weights (BlazeBlock_WT format) into a trainable model.

    This function handles the conversion from MediaPipe's folded-BatchNorm format
    to our trainable BlazeBlock format with explicit BatchNorm layers.

    For loading our own trained checkpoints (which are already in BlazeBlock format),
    use model.load_state_dict() directly instead.

    Args:
        model: Model using BlazeBlock (trainable with BatchNorm)
        weights_path: Path to .pth file with MediaPipe BlazeBlock_WT weights
        strict: Whether to require exact match
        load_detection_heads: Whether to load classifier/regressor heads (default: True).
                             Set to False when fine-tuning for a different detection task.

    Returns:
        missing_keys: Keys in model not found in weights
        unexpected_keys: Keys in weights not found in model
    """
    # Load original state dict (BlazeBlock_WT / MediaPipe format)
    old_state_dict = torch.load(weights_path, map_location='cpu')

    # Convert to trainable BlazeBlock format
    new_state_dict = convert_blazeface_wt_to_trainable(old_state_dict)

    # Optionally exclude detection heads (classifier/regressor)
    # This is useful when fine-tuning for a different detection task
    if not load_detection_heads:
        keys_to_remove = [k for k in new_state_dict.keys()
                         if 'classifier' in k or 'regressor' in k]
        for key in keys_to_remove:
            del new_state_dict[key]

    # Load into model
    result = model.load_state_dict(new_state_dict, strict=strict)

    return result.missing_keys, result.unexpected_keys


# =============================================================================
# Anchor Options (MediaPipe format)
# =============================================================================

anchor_options = {
    "num_layers": 4,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 128,
    "input_size_width": 128,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 16, 16],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}


# =============================================================================
# Model Building Blocks
# =============================================================================

class BlazeBlock(nn.Module):
    """
    BlazeBlock with BatchNormalization for training.
    
    Following vincent1bt/blazeface-tensorflow architecture:
    - DepthwiseConv2D -> BatchNorm -> Conv2D 1x1 -> BatchNorm -> Add -> Activation
    
    This is the default trainable block with explicit BatchNorm layers.
    For loading pretrained MediaPipe weights (which have BatchNorm folded into
    conv weights), use BlazeBlock_WT and the weight conversion utilities.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='relu', skip_proj=False):
        super(BlazeBlock, self).__init__()
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_pad = out_channels - in_channels
        
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        
        # Depthwise convolution (no bias when using BatchNorm)
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if stride == 1 else 0,
            groups=in_channels,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution (no bias when using BatchNorm)
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Optional skip projection (matches BlazeBlock interface)
        if skip_proj:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_proj = None
        
        # Activation (matches BlazeBlock interface)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s" % act)
    
    def forward(self, x):
        # Handle stride=2 padding (TFLite compatibility, matches BlazeBlock)
        if self.stride == 2:
            if self.kernel_size == 3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x_skip = self.max_pool(x)
        else:
            h = x
            x_skip = x
        
        # Depthwise + Pointwise convolutions with BatchNorm
        h = self.dw_conv(h)
        h = self.bn1(h)
        h = self.pw_conv(h)
        h = self.bn2(h)
        
        # Skip connection (matches BlazeBlock logic)
        if self.skip_proj is not None:
            x_skip = self.skip_proj(x_skip)
        elif self.channel_pad > 0:
            x_skip = F.pad(x_skip, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        
        return self.act(h + x_skip)


class BlazeBlock_WT(nn.Module):
    """
    BlazeBlock for Weight Transfer - used for loading pretrained MediaPipe weights.
    
    This version has BatchNorm folded into conv weights (no explicit BN layers),
    matching the format of pretrained .pth files from MediaPipe conversion.
    
    Use the weight conversion utilities (convert_blazeface_to_bn, load_pretrained_to_bn_model)
    to transfer weights from this format to the trainable BlazeBlock format.
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
        #theta = np.arctan2(y0-y1, x0-x1) - self.theta0
        theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        return xc, yc, scale, theta







