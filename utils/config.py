"""
Configuration constants for BlazeFace detection.
"""

# Anchor configuration
SMALL_GRID_SIZE = 16
BIG_GRID_SIZE = 8
SMALL_ANCHORS_PER_CELL = 2
BIG_ANCHORS_PER_CELL = 6
TOTAL_ANCHORS = 896  # 16*16*2 + 8*8*6

# Default paths
DEFAULT_WEIGHTS_PATH = "model_weights/blazeface.pth"
DEFAULT_DATA_ROOT = "data/raw/blazeface"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Model parameters
DEFAULT_INPUT_SIZE = 128
DEFAULT_DETECTION_THRESHOLD = 0.9
DEFAULT_TRAIN_THRESHOLD = 0.3
