# config.py
"""
Configuration for BlazeEar ear detector.

Similar to BlazeFace configuration but tuned for ear detection:
- Ear-specific anchor aspect ratios (taller than wide, ~1.4-2.4 h/w ratio)
- Anchor sizes derived from analysis of 17,447 ear bounding boxes
"""

# =============================================================================
# BlazeEar Configuration (128x128 input, like BlazeFace front model)
# =============================================================================
cfg_blazeear = {
    'name': 'BlazeEar',
    'min_dim': 128,
    'feature_maps': [[16, 16], [8, 8]],  # Spatial dims at output layers
    'steps': [8, 16],                     # Receptive field steps (128/16=8, 128/8=16)
    
    # Anchor configuration (derived from ear bbox analysis)
    # Based on data analysis of 17,447 ear bounding boxes:
    #   - Width:  10th=0.015, 25th=0.027, 50th=0.049, 75th=0.086, 90th=0.130
    #   - Height: 10th=0.033, 25th=0.059, 50th=0.101, 75th=0.161, 90th=0.225
    #   - Aspect ratio (h/w): median=1.85, range 1.4-2.4
    'anchor_config_16': {
        'base_sizes': [0.04, 0.08, 0.12],   # Cover 10th-75th percentile (smaller ears)
        'aspect_ratios': [1.4, 1.85, 2.4],  # 25th, 50th, 75th percentile h/w
    },
    'anchor_config_8': {
        'base_sizes': [0.15, 0.25, 0.35],   # Cover 75th-95th percentile (larger ears)
        'aspect_ratios': [1.4, 1.85, 2.4],
    },
    
    # Variance-based encoding (standard SSD/BlazeFace approach)
    # variance[0]: for center offsets (dx, dy) - smaller = larger gradients
    # variance[1]: for log-scale size (dw, dh) - controls size sensitivity
    'variance': [0.1, 0.2],
    
    # Matching thresholds for anchor-GT assignment
    'pos_iou_threshold': 0.5,   # IoU >= this -> positive anchor
    'neg_iou_threshold': 0.4,   # IoU < this -> negative anchor
    'min_anchor_iou': 0.3,      # Reject GT boxes where best anchor IoU < this
    
    # Clipping (whether to clip anchors to [0, 1])
    'clip': False,
    
    # Number of classes (1 for ear only)
    'num_classes': 1,
}

# Legacy constants for backward compatibility
# These are extracted from cfg_blazeear for direct access
ANCHOR_CONFIG_16 = cfg_blazeear['anchor_config_16']
ANCHOR_CONFIG_8 = cfg_blazeear['anchor_config_8']
VARIANCE = cfg_blazeear['variance']
SCALE_FACTOR = 128.0  # Legacy - kept for backward compatibility

MATCHING_CONFIG = {
    'pos_iou_threshold': cfg_blazeear['pos_iou_threshold'],
    'neg_iou_threshold': cfg_blazeear['neg_iou_threshold'],
    'min_anchor_iou': cfg_blazeear['min_anchor_iou'],
}


def get_num_anchors_per_cell(config: dict) -> int:
    """Get number of anchors per grid cell for a config."""
    return len(config['base_sizes']) * len(config['aspect_ratios'])
