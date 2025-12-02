# blazeface.py
"""
BlazeFace model architecture for face detection.

BlazeFace Front model (128x128 input) with 6 facial keypoints.
Efficient single-shot face detector optimized for mobile/edge deployment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg_blazeface


class BlazeBlock(nn.Module):
    """
    BlazeBlock - Efficient building block for BlazeFace.
    
    Uses depthwise separable convolutions with skip connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(BlazeBlock, self).__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels
        
        # Consistent padding logic
        padding = (kernel_size - 1) // 2
        
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        
        self.convs = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.stride == 2:
            h = self.max_pool(x)
            if self.channel_pad > 0:
                h = F.pad(h, (0, 0, 0, 0, 0, self.channel_pad))
            return self.act(self.convs(x) + h)
        else:
            return self.act(self.convs(x) + x)


class BlazeFaceBackbone(nn.Module):
    """
    BlazeFace backbone - feature extraction network.
    
    Two-stage feature extraction producing 16x16 and 8x8 feature maps.
    """
    def __init__(self):
        super(BlazeFaceBackbone, self).__init__()
        
        # Stage 1: 128x128 -> 16x16 (stride 8)
        self.backbone1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48)
        )
        
        # Stage 2: 16x16 -> 8x8 (stride 16)
        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 24, stride=2),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48)
        )
    
    def forward(self, x):
        """
        Returns:
            h1: 16x16 feature map (48 channels)
            h2: 8x8 feature map (48 channels)
        """
        h1 = self.backbone1(x)  # 16x16
        h2 = self.backbone2(h1)  # 8x8
        return h1, h2


class BlazeFace(nn.Module):
    """
    BlazeFace face detector.
    
    Single-shot detector with dual-scale predictions (16x16 and 8x8).
    Outputs classification scores and box/keypoint regression.
    
    Args:
        input_size: Input image size (default 128)
        num_keypoints: Number of keypoints to predict (default 6)
        cfg: Optional config dict (defaults to cfg_blazeface)
    """
    def __init__(self, input_size: int = 128, num_keypoints: int = 6, cfg=None):
        super(BlazeFace, self).__init__()
        
        self.input_size = input_size
        self.num_keypoints = num_keypoints
        self.cfg = cfg if cfg is not None else cfg_blazeface
        
        self.num_classes = 1  # Face vs Background
        self.num_anchors = [2, 6]  # Anchors per pixel for layer 1 and 2
        
        # Backbone
        self.backbone = BlazeFaceBackbone()
        
        # Detection Heads
        # Regression output: 4 (box) + num_keypoints * 2 (x,y per keypoint)
        reg_output_size = 4 + self.num_keypoints * 2
        
        # Layer 1 (16x16) -> 2 anchors per pixel
        self.classifier_1 = nn.Conv2d(48, self.num_anchors[0] * self.num_classes, kernel_size=1)
        self.regressor_1 = nn.Conv2d(48, self.num_anchors[0] * reg_output_size, kernel_size=1)
        
        # Layer 2 (8x8) -> 6 anchors per pixel
        self.classifier_2 = nn.Conv2d(48, self.num_anchors[1] * self.num_classes, kernel_size=1)
        self.regressor_2 = nn.Conv2d(48, self.num_anchors[1] * reg_output_size, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            conf: Classification scores (B, num_anchors, 1)
            loc: Box + keypoint regression (B, num_anchors, 4 + num_keypoints*2)
        """
        h1, h2 = self.backbone(x)
        
        reg_size = 4 + self.num_keypoints * 2
        
        # Compute predictions - reshape to (B, num_anchors, channels)
        c1 = self.classifier_1(h1).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        r1 = self.regressor_1(h1).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, reg_size)
        
        c2 = self.classifier_2(h2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        r2 = self.regressor_2(h2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, reg_size)

        # Concatenate predictions from both layers
        conf = torch.cat([c1, c2], dim=1)  # (B, 896, 1)
        loc = torch.cat([r1, r2], dim=1)   # (B, 896, 16) with 6 keypoints
        
        return conf, loc
    
    @property
    def num_anchors_total(self):
        """Total number of anchors (16*16*2 + 8*8*6 = 896)."""
        return 16 * 16 * self.num_anchors[0] + 8 * 8 * self.num_anchors[1]