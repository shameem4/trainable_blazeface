"""
BlazeEar Model - Ear detection model based on BlazeFace architecture.

Uses the BlazeFace backbone from MediaPipe for efficient ear detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class BlazeBlock(nn.Module):
    """BlazeFace building block with depthwise separable convolution.
    
    Note: No BatchNorm - weights are folded into conv in pretrained BlazeFace.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        # Match original BlazeFace structure (no BatchNorm - folded into weights)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)


class FinalBlazeBlock(nn.Module):
    """Final BlazeFace block with stride 2."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        # Match original BlazeFace structure (no BatchNorm)
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, 2, 0, groups=channels, bias=True),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (0, 2, 0, 2), "constant", 0)
        return self.act(self.convs(h))


class BlazeEarBackbone(nn.Module):
    """
    BlazeFace-inspired backbone for ear detection.
    
    Uses the front-facing BlazeFace architecture which expects 128x128 input.
    Outputs features at two scales for multi-scale detection.
    
    Note: No BatchNorm layers to match pretrained BlazeFace weights.
    """
    
    def __init__(self):
        super().__init__()
        
        # First stage backbone (outputs 16x16 feature map)
        # Match original BlazeFace structure exactly (no BatchNorm)
        self.backbone1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 0, bias=True),
            nn.ReLU(inplace=True),
            
            BlazeBlock(24, 24),
            BlazeBlock(24, 28),
            BlazeBlock(28, 32, stride=2),
            BlazeBlock(32, 36),
            BlazeBlock(36, 42),
            BlazeBlock(42, 48, stride=2),
            BlazeBlock(48, 56),
            BlazeBlock(56, 64),
            BlazeBlock(64, 72),
            BlazeBlock(72, 80),
            BlazeBlock(80, 88),
        )
        
        # Second stage backbone (outputs 8x8 feature map)
        self.backbone2 = nn.Sequential(
            BlazeBlock(88, 96, stride=2),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
            BlazeBlock(96, 96),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pad input (BlazeFace specific padding)
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        # First stage: 16x16 features
        feat1 = self.backbone1(x)  # (B, 88, 16, 16)
        
        # Second stage: 8x8 features  
        feat2 = self.backbone2(feat1)  # (B, 96, 8, 8)
        
        return feat1, feat2


class BlazeEar(nn.Module):
    """
    BlazeEar - Ear detection model based on BlazeFace architecture.
    
    Outputs bounding box predictions at two scales (16x16 and 8x8).
    Uses SSD-style anchor-based detection.
    
    Args:
        num_anchors_16: Number of anchors per location at 16x16 scale
        num_anchors_8: Number of anchors per location at 8x8 scale
        input_size: Input image size (default 128 for BlazeFace compatibility)
        anchor_config_path: Path to custom anchor config (.npy file from create_detector_anchors.py)
    """
    
    def __init__(
        self, 
        num_anchors_16: int = 2,
        num_anchors_8: int = 6,
        input_size: int = 128,
        anchor_config_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_anchors_16 = num_anchors_16
        self.num_anchors_8 = num_anchors_8
        self.num_classes = 1  # Ear only
        self.anchor_config_path = anchor_config_path
        
        # Load custom anchor sizes if provided
        self.anchor_sizes_16, self.anchor_sizes_8 = self._load_anchor_config(anchor_config_path)
        
        # Backbone
        self.backbone = BlazeEarBackbone()
        
        # Detection heads at 16x16 scale (from backbone1, 88 channels)
        self.classifier_16 = nn.Conv2d(88, num_anchors_16 * 1, 1)  # 1 class score per anchor
        self.regressor_16 = nn.Conv2d(88, num_anchors_16 * 4, 1)   # 4 bbox coords per anchor
        
        # Detection heads at 8x8 scale (from backbone2, 96 channels)
        self.classifier_8 = nn.Conv2d(96, num_anchors_8 * 1, 1)
        self.regressor_8 = nn.Conv2d(96, num_anchors_8 * 4, 1)
        
        # Generate anchors
        self.register_buffer('anchors', self._generate_anchors())
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection head weights."""
        for m in [self.classifier_16, self.classifier_8]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, -4.6)  # Initialize to low confidence
            
        for m in [self.regressor_16, self.regressor_8]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
    
    def _load_anchor_config(self, config_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load custom anchor sizes from config file.
        
        Args:
            config_path: Path to anchor config .npy file
            
        Returns:
            Tuple of (anchors_16, anchors_8) arrays with [w, h] per anchor
        """
        if config_path is None:
            # Use default anchor sizes
            default_16 = np.array([[0.1, 0.1]] * self.num_anchors_16)
            default_8 = np.array([[0.2, 0.2]] * self.num_anchors_8)
            return default_16, default_8
        
        try:
            config = np.load(config_path, allow_pickle=True).item()
            anchors_16 = np.array(config['anchors_16'])
            anchors_8 = np.array(config['anchors_8'])
            
            # Validate anchor counts match
            if len(anchors_16) != self.num_anchors_16:
                print(f"Warning: anchor config has {len(anchors_16)} anchors for 16x16, "
                      f"but model expects {self.num_anchors_16}. Padding/truncating.")
                anchors_16 = self._adjust_anchor_count(anchors_16, self.num_anchors_16, 0.1)
            
            if len(anchors_8) != self.num_anchors_8:
                print(f"Warning: anchor config has {len(anchors_8)} anchors for 8x8, "
                      f"but model expects {self.num_anchors_8}. Padding/truncating.")
                anchors_8 = self._adjust_anchor_count(anchors_8, self.num_anchors_8, 0.2)
            
            print(f"Loaded custom anchors from {config_path}")
            print(f"  16x16 anchors: {anchors_16.tolist()}")
            print(f"  8x8 anchors: {anchors_8.tolist()}")
            
            return anchors_16, anchors_8
            
        except Exception as e:
            print(f"Warning: Could not load anchor config from {config_path}: {e}")
            print("Using default anchor sizes.")
            default_16 = np.array([[0.1, 0.1]] * self.num_anchors_16)
            default_8 = np.array([[0.2, 0.2]] * self.num_anchors_8)
            return default_16, default_8
    
    def _adjust_anchor_count(self, anchors: np.ndarray, target_count: int, default_size: float) -> np.ndarray:
        """Adjust anchor array to target count by padding or truncating."""
        if len(anchors) < target_count:
            # Pad with default anchors
            padding = np.array([[default_size, default_size]] * (target_count - len(anchors)))
            return np.vstack([anchors, padding])
        else:
            # Truncate
            return anchors[:target_count]
    
    def _generate_anchors(self) -> torch.Tensor:
        """
        Generate SSD-style anchors at two scales.
        
        Uses custom anchor sizes if loaded from config, otherwise uses defaults.
        """
        anchors = []
        
        # 16x16 grid anchors
        for y in range(16):
            for x in range(16):
                cx = (x + 0.5) / 16
                cy = (y + 0.5) / 16
                for i in range(self.num_anchors_16):
                    # Use custom anchor sizes
                    w, h = self.anchor_sizes_16[i]
                    anchors.append([cx, cy, float(w), float(h)])
        
        # 8x8 grid anchors  
        for y in range(8):
            for x in range(8):
                cx = (x + 0.5) / 8
                cy = (y + 0.5) / 8
                for i in range(self.num_anchors_8):
                    # Use custom anchor sizes
                    w, h = self.anchor_sizes_8[i]
                    anchors.append([cx, cy, float(w), float(h)])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 128, 128) normalized to [-1, 1]
            
        Returns:
            Dictionary with:
                - 'box_regression': (B, num_anchors, 4) bbox deltas
                - 'classification': (B, num_anchors, 1) class scores
        """
        batch_size = x.shape[0]
        
        # Get features
        feat1, feat2 = self.backbone(x)
        
        # 16x16 predictions
        cls_16 = self.classifier_16(feat1)  # (B, num_anchors_16, 16, 16)
        reg_16 = self.regressor_16(feat1)   # (B, num_anchors_16*4, 16, 16)
        
        # 8x8 predictions
        cls_8 = self.classifier_8(feat2)    # (B, num_anchors_8, 8, 8)
        reg_8 = self.regressor_8(feat2)     # (B, num_anchors_8*4, 8, 8)
        
        # Reshape to (B, num_anchors, C)
        cls_16 = cls_16.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        reg_16 = reg_16.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        cls_8 = cls_8.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)
        reg_8 = reg_8.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        
        # Concatenate scales
        classification = torch.cat([cls_16, cls_8], dim=1)  # (B, 512+384, 1)
        box_regression = torch.cat([reg_16, reg_8], dim=1)  # (B, 512+384, 4)
        
        return {
            'classification': classification,
            'box_regression': box_regression,
        }
    
    def decode_boxes(
        self,
        box_regression: torch.Tensor,
        anchors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode box regression predictions to actual boxes.
        
        Args:
            box_regression: (B, num_anchors, 4) predicted deltas
            anchors: (num_anchors, 4) anchor boxes [cx, cy, w, h]
            
        Returns:
            (B, num_anchors, 4) decoded boxes in [x1, y1, x2, y2] format
        """
        if anchors is None:
            anchors = self.anchors
            
        # Decode center and size
        pred_cx = box_regression[..., 0] * anchors[:, 2] + anchors[:, 0]
        pred_cy = box_regression[..., 1] * anchors[:, 3] + anchors[:, 1]
        pred_w = torch.exp(box_regression[..., 2]) * anchors[:, 2]
        pred_h = torch.exp(box_regression[..., 3]) * anchors[:, 3]
        
        # Convert to corner format
        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run inference and return detections.
        
        Args:
            x: Input tensor (B, 3, 128, 128)
            score_threshold: Minimum score for detection
            nms_threshold: NMS IoU threshold
            
        Returns:
            List of dicts per image with 'boxes', 'scores'
        """
        outputs = self.forward(x)
        
        scores = torch.sigmoid(outputs['classification'])  # (B, N, 1)
        boxes = self.decode_boxes(outputs['box_regression'])  # (B, N, 4)
        
        results = []
        for i in range(x.shape[0]):
            score = scores[i, :, 0]
            box = boxes[i]
            
            # Filter by score
            mask = score > score_threshold
            score = score[mask]
            box = box[mask]
            
            # NMS
            if len(score) > 0:
                from torchvision.ops import nms
                keep = nms(box, score, nms_threshold)
                score = score[keep]
                box = box[keep]
            
            results.append({
                'boxes': box,
                'scores': score,
            })
        
        return results


def create_blazeear(pretrained_blazeface_path: Optional[str] = None) -> BlazeEar:
    """
    Create BlazeEar model, optionally loading BlazeFace backbone weights.
    
    Args:
        pretrained_blazeface_path: Path to pretrained BlazeFace weights
        
    Returns:
        BlazeEar model
    """
    model = BlazeEar()
    
    if pretrained_blazeface_path is not None:
        # Load BlazeFace weights and transfer backbone
        blazeface_state = torch.load(pretrained_blazeface_path, map_location='cpu')
        
        # Map BlazeFace backbone weights to BlazeEar backbone
        # BlazeFace uses backbone1.X and backbone2.X
        # BlazeEar uses backbone.backbone1.X and backbone.backbone2.X
        backbone_state = {}
        for key, value in blazeface_state.items():
            if key.startswith('backbone1.') or key.startswith('backbone2.'):
                new_key = f'backbone.{key}'
                backbone_state[new_key] = value
        
        # Load with strict=False since detection heads are different
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f"Loaded backbone weights from {pretrained_blazeface_path}")
        print(f"  Loaded {len(backbone_state)} backbone weights")
        print(f"  Missing keys (detection heads): {len(missing)}")
    
    return model
    return model
