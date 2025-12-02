# blazeface_anchors.py
import torch
from itertools import product
from math import sqrt

class AnchorGenerator:
    def __init__(self, cfg):
        self.min_dim = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']

    def forward(self):
        priors = []
        for k, f in enumerate(self.feature_maps): # Loop over layer 1 and 2
            # f[0] = height, f[1] = width (e.g., 16x16)
            for i, j in product(range(f[0]), range(f[1])):
                f_k = self.min_dim / self.steps[k]
                
                # Unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Small Sized Square Box
                s_k = self.min_sizes[k][0] / self.min_dim
                priors.append([cx, cy, s_k, s_k])

                # Aspect Ratio boxes (if any) or extra sizes
                # Note: BlazeFace specific logic - it usually just adds more fixed sizes
                # The paper/implementation differs here. 
                # Following standard implementation which creates N boxes per pixel:
                for size in self.min_sizes[k][1:]:
                    s_k_prime = size / self.min_dim
                    priors.append([cx, cy, s_k_prime, s_k_prime])
                    
        output = torch.Tensor(priors)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output