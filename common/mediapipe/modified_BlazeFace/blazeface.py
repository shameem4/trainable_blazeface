# blazeface.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BlazeBlock(nn.Module):
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

class BlazeFace(nn.Module):
    def __init__(self, cfg):
        super(BlazeFace, self).__init__()
        self.cfg = cfg
        self.num_classes = 1 # Face vs Bg
        self.num_anchors = [2, 6] # Anchors per pixel for layer 1 and 2
        
        # --- Backbone ---
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
        
        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 24, stride=2),
            BlazeBlock(24, 24),
            BlazeBlock(24, 24),
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48),
            BlazeBlock(48, 48)
        )

        # --- Heads ---
        # Layer 1 (16x16) -> 2 anchors
        self.classifier_1 = nn.Conv2d(48, self.num_anchors[0] * self.num_classes, kernel_size=1)
        self.regressor_1 = nn.Conv2d(48, self.num_anchors[0] * (4 + cfg['num_keypoints']*2), kernel_size=1)
        
        # Layer 2 (8x8) -> 6 anchors
        self.classifier_2 = nn.Conv2d(48, self.num_anchors[1] * self.num_classes, kernel_size=1)
        self.regressor_2 = nn.Conv2d(48, self.num_anchors[1] * (4 + cfg['num_keypoints']*2), kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h1 = self.backbone1(x) # 16x16
        h2 = self.backbone2(h1) # 8x8
        
        # Compute predictions
        c1 = self.classifier_1(h1).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        r1 = self.regressor_1(h1).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4 + self.cfg['num_keypoints']*2)
        
        c2 = self.classifier_2(h2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        r2 = self.regressor_2(h2).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4 + self.cfg['num_keypoints']*2)

        return torch.cat([c1, c2], dim=1), torch.cat([r1, r2], dim=1)