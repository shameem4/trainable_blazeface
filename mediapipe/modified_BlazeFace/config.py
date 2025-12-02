# config.py

# Standard BlazeFace Front (128x128) Configuration
cfg_blazeface_front = {
    'name': 'BlazeFaceFront',
    'min_dim': 128,
    'feature_maps': [[16, 16], [8, 8]], # Spatial dims at output layers
    'steps': [8, 16],                   # Receptive field steps (128/16=8, 128/8=16)
    'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]], # Anchor sizes
    'aspect_ratios': [[1], [1]], 
    'variance': [0.1, 0.2],
    'clip': False,
    'num_keypoints': 6
}

# You can add 'cfg_blazeface_back' here easily following the same pattern