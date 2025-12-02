"""
Generate BlazeFace-style anchors for ear detection.

BlazeFace uses fixed unit anchors (w=h=1.0) at two scales:
- 16x16 grid with 2 anchors per cell = 512 anchors
- 8x8 grid with 6 anchors per cell = 384 anchors
- Total: 896 anchors

The network learns to predict offsets relative to these fixed anchor positions.
Scale factors (128 for front model, 256 for back) handle the coordinate scaling.

Usage:
    python shared/data_processing/create_blazeface_anchors.py
    
    # For back model (256x256 input)
    python shared/data_processing/create_blazeface_anchors.py --back_model
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent.parent))


def generate_blazeface_anchors(
    input_size: int = 128,
    strides: list = None,
    anchors_per_layer: list = None,
) -> np.ndarray:
    """
    Generate BlazeFace-style anchors.
    
    BlazeFace anchor generation from MediaPipe's SsdAnchorsCalculator:
    - Uses fixed unit anchors (w=h=1.0) when fixed_anchor_size=True
    - Grid centers at (x + 0.5) / feature_map_size
    - Order: for each y, iterate through x, then through anchors at that position
    
    Args:
        input_size: Input image size (128 for front, 256 for back)
        strides: Feature map strides. Default [8, 16] for front model.
        anchors_per_layer: Number of anchors per cell at each stride.
                          Default [2, 6] for front model.
    
    Returns:
        anchors: (896, 4) array of [cx, cy, 1.0, 1.0] anchors
    """
    if strides is None:
        # Front model: stride 8 (16x16) and stride 16 (8x8)
        strides = [8, 16]
    
    if anchors_per_layer is None:
        # Front model: 2 anchors at 16x16, 6 anchors at 8x8
        anchors_per_layer = [2, 6]
    
    anchors = []
    
    for stride, num_anchors in zip(strides, anchors_per_layer):
        feature_map_size = input_size // stride
        
        # BlazeFace order: y outer loop, x inner loop
        for y in range(feature_map_size):
            for x in range(feature_map_size):
                # Center coordinates (offset by 0.5)
                cx = (x + 0.5) / feature_map_size
                cy = (y + 0.5) / feature_map_size
                
                # Add num_anchors copies at this position
                # BlazeFace uses unit anchors (w=h=1.0)
                for _ in range(num_anchors):
                    anchors.append([cx, cy, 1.0, 1.0])
    
    return np.array(anchors, dtype=np.float32)


def generate_anchors_mediapipe_style(options: dict) -> np.ndarray:
    """
    Generate anchors using MediaPipe's exact algorithm.
    
    This is a direct port of MediaPipe's ssd_anchors_calculator.cc
    for maximum compatibility.
    
    Args:
        options: Dictionary with anchor generation options
        
    Returns:
        anchors: (N, 4) array of [cx, cy, w, h] anchors
    """
    def calculate_scale(min_scale, max_scale, stride_index, num_strides):
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)
    
    strides = options["strides"]
    strides_size = len(strides)
    
    anchors = []
    layer_id = 0
    
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        
        # For same strides, merge anchors
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size and 
               strides[last_same_stride_layer] == strides[layer_id]):
            
            scale = calculate_scale(
                options["min_scale"],
                options["max_scale"],
                last_same_stride_layer,
                strides_size
            )
            
            if last_same_stride_layer == 0 and options.get("reduce_boxes_in_lowest_layer", False):
                aspect_ratios.extend([1.0, 2.0, 0.5])
                scales.extend([0.1, scale, scale])
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)
                
                if options.get("interpolated_scale_aspect_ratio", 0) > 0:
                    if last_same_stride_layer == strides_size - 1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(
                            options["min_scale"],
                            options["max_scale"],
                            last_same_stride_layer + 1,
                            strides_size
                        )
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])
            
            last_same_stride_layer += 1
        
        for i in range(len(aspect_ratios)):
            ratio_sqrt = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrt)
            anchor_width.append(scales[i] * ratio_sqrt)
        
        stride = strides[layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))
        
        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height
                    
                    if options.get("fixed_anchor_size", False):
                        # BlazeFace style: unit anchors
                        anchors.append([x_center, y_center, 1.0, 1.0])
                    else:
                        # SSD style: sized anchors
                        anchors.append([x_center, y_center, 
                                       anchor_width[anchor_id], 
                                       anchor_height[anchor_id]])
        
        layer_id = last_same_stride_layer
    
    return np.array(anchors, dtype=np.float32)


def get_blazeface_front_options():
    """Get anchor options for BlazeFace front model (128x128)."""
    return {
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


def get_blazeface_back_options():
    """Get anchor options for BlazeFace back model (256x256)."""
    return {
        "num_layers": 4,
        "min_scale": 0.15625,
        "max_scale": 0.75,
        "input_size_height": 256,
        "input_size_width": 256,
        "anchor_offset_x": 0.5,
        "anchor_offset_y": 0.5,
        "strides": [16, 32, 32, 32],
        "aspect_ratios": [1.0],
        "reduce_boxes_in_lowest_layer": False,
        "interpolated_scale_aspect_ratio": 1.0,
        "fixed_anchor_size": True,
    }


def verify_against_blazeface(generated: np.ndarray, reference_path: str) -> bool:
    """Verify generated anchors match BlazeFace reference."""
    if not os.path.exists(reference_path):
        print(f"Reference file not found: {reference_path}")
        return False
    
    reference = np.load(reference_path)
    
    if generated.shape != reference.shape:
        print(f"Shape mismatch: {generated.shape} vs {reference.shape}")
        return False
    
    max_diff = np.abs(generated - reference).max()
    if max_diff > 1e-5:
        print(f"Value mismatch: max difference = {max_diff}")
        return False
    
    print(f"✓ Generated anchors match reference exactly!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate BlazeFace-style anchors for ear detection"
    )
    parser.add_argument(
        "--back_model",
        action="store_true",
        help="Generate anchors for back model (256x256) instead of front (128x128)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ear_detector/data/preprocessed",
        help="Output directory for anchor files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify against BlazeFace reference anchors"
    )
    args = parser.parse_args()
    
    # Get appropriate options
    if args.back_model:
        options = get_blazeface_back_options()
        suffix = "_back"
        reference_path = "common/mediapipe/BlazeFace/anchorsback.npy"
    else:
        options = get_blazeface_front_options()
        suffix = ""
        reference_path = "common/mediapipe/BlazeFace/anchors.npy"
    
    print(f"Generating BlazeFace-style anchors...")
    print(f"  Input size: {options['input_size_width']}x{options['input_size_height']}")
    print(f"  Strides: {options['strides']}")
    print(f"  Fixed anchor size: {options['fixed_anchor_size']}")
    
    # Generate anchors using MediaPipe algorithm
    anchors = generate_anchors_mediapipe_style(options)
    
    print(f"\nGenerated {len(anchors)} anchors")
    print(f"  Shape: {anchors.shape}")
    print(f"  Format: [cx, cy, w, h]")
    print(f"  First anchor: {anchors[0]}")
    print(f"  Last anchor: {anchors[-1]}")
    
    # Verify layout
    # 16x16 grid with 2 anchors = 512 (or 8x8 with 2 for back model)
    # 8x8 grid with 6 anchors = 384 (or 4x4 with 6 for back model)
    if not args.back_model:
        assert len(anchors) == 896, f"Expected 896 anchors, got {len(anchors)}"
        print(f"\n  First scale (16x16, 2 per cell): anchors 0-511")
        print(f"  Second scale (8x8, 6 per cell): anchors 512-895")
    
    # Verify against reference if requested
    if args.verify:
        print(f"\nVerifying against: {reference_path}")
        verify_against_blazeface(anchors, reference_path)
    
    # Save anchors
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"anchors_ear{suffix}.npy")
    np.save(output_path, anchors)
    print(f"\n✓ Saved anchors to: {output_path}")
    
    # Also show how to use in model
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
The anchors are in BlazeFace format: [cx, cy, 1.0, 1.0]
- cx, cy: Grid cell center in normalized coordinates (0-1)  
- w, h: Always 1.0 (unit anchors)

During training, encode targets as:
    dx = (gt_cx - anchor_cx) * scale  # scale = 128 for front model
    dy = (gt_cy - anchor_cy) * scale
    dw = gt_w * scale
    dh = gt_h * scale

During inference, decode predictions as:
    pred_cx = raw[0] / scale + anchor_cx
    pred_cy = raw[1] / scale + anchor_cy  
    pred_w = raw[2] / scale
    pred_h = raw[3] / scale

IoU matching threshold should be LOW (0.1) because unit anchors
have low IoU with small objects.
""")


if __name__ == "__main__":
    main()
