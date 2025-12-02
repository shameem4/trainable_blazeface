# config.py
"""
Configuration for BlazeFace face detector.

Standard BlazeFace Front model (128x128 input) with 6 facial keypoints:
- Right eye, Left eye, Nose tip, Mouth center, Right ear, Left ear
"""

# =============================================================================
# BlazeFace Front Configuration (128x128 input)
# =============================================================================
cfg_blazeface = {
    'name': 'BlazeFaceFront',
    'min_dim': 128,
    'feature_maps': [[16, 16], [8, 8]],  # Spatial dims at output layers
    'steps': [8, 16],                     # Receptive field steps (128/16=8, 128/8=16)
    
    # Anchor configuration
    # Layer 1 (16x16): 2 anchors per pixel = 512 anchors
    # Layer 2 (8x8): 6 anchors per pixel = 384 anchors
    # Total: 896 anchors
    'anchor_config_16': {
        'base_sizes': [16, 24],  # Pixel sizes (will be normalized by 128)
    },
    'anchor_config_8': {
        'base_sizes': [32, 48, 64, 80, 96, 128],  # Larger faces
    },
    
    # Variance-based encoding (standard SSD/BlazeFace approach)
    'variance': [0.1, 0.2],
    
    # Matching thresholds for anchor-GT assignment
    'pos_iou_threshold': 0.35,  # IoU >= this -> positive anchor
    'neg_iou_threshold': 0.35,  # IoU < this -> negative anchor
    'min_anchor_iou': 0.2,      # Reject GT boxes where best anchor IoU < this
    
    # Clipping
    'clip': False,
    
    # Number of keypoints (6 for BlazeFace: eyes, nose, mouth, ears)
    'num_keypoints': 6,
    
    # Number of classes (1 for face only)
    'num_classes': 1,
}

# Legacy alias for backward compatibility
cfg_blazeface_front = cfg_blazeface

# =============================================================================
# Matching Configuration (exported for use by other modules)
# =============================================================================
MATCHING_CONFIG = {
    'pos_iou_threshold': cfg_blazeface['pos_iou_threshold'],
    'neg_iou_threshold': cfg_blazeface['neg_iou_threshold'],
    'min_anchor_iou': cfg_blazeface['min_anchor_iou'],
    'variance': cfg_blazeface['variance'],
}