"""
Anchor Generation for Ear Detector.

Analyzes bounding box distributions in the training data and generates
optimal anchors using K-means clustering.

Usage:
    python shared/data_processing/create_detector_anchors.py
    
    # With custom paths
    python shared/data_processing/create_detector_anchors.py \
        --train_metadata data/preprocessed/train_detector.npy \
        --val_metadata data/preprocessed/val_detector.npy \
        --output_dir data/preprocessed \
        --num_anchors_16 2 \
        --num_anchors_8 6

Output:
    - detector_anchors.npy: Anchor configurations for ear detector
    - detector_anchor_analysis.png: Visualization of bbox distributions and anchors
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# Add parent directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent.parent))


def load_bboxes(metadata_path: str, normalize: bool = True) -> np.ndarray:
    """
    Load bounding boxes from metadata file and normalize by image size.
    
    Args:
        metadata_path: Path to .npy metadata file
        normalize: Whether to normalize bboxes to 0-1 range (relative to image size)
        
    Returns:
        Array of bboxes in [x, y, w, h] format (normalized 0-1 if normalize=True)
    """
    if not os.path.exists(metadata_path):
        print(f"Warning: {metadata_path} not found")
        return np.array([])
    
    metadata = np.load(metadata_path, allow_pickle=True).item()
    image_paths = metadata.get('image_paths', [])
    all_bboxes = []
    
    for idx, bboxes in enumerate(metadata['bboxes']):
        # Handle list of bboxes per image
        if isinstance(bboxes, (list, np.ndarray)):
            # Get image dimensions for normalization
            img_width, img_height = None, None
            if normalize and idx < len(image_paths):
                try:
                    img_path = image_paths[idx]
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    img.close()
                except Exception as e:
                    # Fall back to typical image size if we can't load
                    pass
            
            for bbox in bboxes:
                bbox = np.array(bbox).flatten()
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    
                    # Normalize by image dimensions
                    if normalize and img_width and img_height:
                        w = w / img_width
                        h = h / img_height
                    
                    # Skip invalid bboxes
                    if w > 0 and h > 0:
                        all_bboxes.append([x, y, w, h])
    
    return np.array(all_bboxes) if all_bboxes else np.array([])


def convert_to_wh(bboxes: np.ndarray, format: str = 'xywh') -> np.ndarray:
    """
    Convert bboxes to width-height pairs.
    
    Args:
        bboxes: Array of bboxes
        format: 'xywh' or 'xyxy'
        
    Returns:
        Array of [width, height] pairs
    """
    if len(bboxes) == 0:
        return np.array([])
    
    if format == 'xywh':
        # x, y, w, h -> w, h
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
    else:
        # x1, y1, x2, y2 -> w, h
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
    
    return np.stack([widths, heights], axis=1)


def iou_distance(boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Compute 1 - IoU as distance metric for K-means.
    
    Args:
        boxes: (N, 2) array of [w, h]
        clusters: (K, 2) array of cluster centers [w, h]
        
    Returns:
        (N, K) distance matrix
    """
    n = boxes.shape[0]
    k = clusters.shape[0]
    
    # Compute intersection
    box_area = boxes[:, 0] * boxes[:, 1]  # (N,)
    cluster_area = clusters[:, 0] * clusters[:, 1]  # (K,)
    
    # Min of widths and heights for intersection
    inter_w = np.minimum(boxes[:, 0:1], clusters[:, 0:1].T)  # (N, K)
    inter_h = np.minimum(boxes[:, 1:2], clusters[:, 1:2].T)  # (N, K)
    inter_area = inter_w * inter_h
    
    # Union
    union_area = box_area[:, np.newaxis] + cluster_area[np.newaxis, :] - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return 1 - iou


def kmeans_anchors(
    wh_data: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Run K-means clustering on width-height pairs using IoU distance.
    
    Args:
        wh_data: (N, 2) array of [width, height] pairs
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        random_state: Random seed
        
    Returns:
        Tuple of (cluster_centers, avg_iou)
    """
    if len(wh_data) < n_clusters:
        print(f"Warning: Only {len(wh_data)} samples, using {len(wh_data)} clusters")
        n_clusters = len(wh_data)
    
    # Initialize with K-means++
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=max_iter,
        random_state=random_state,
        n_init=10,
    )
    
    kmeans.fit(wh_data)
    clusters = kmeans.cluster_centers_
    
    # Compute average IoU
    distances = iou_distance(wh_data, clusters)
    best_ious = 1 - distances.min(axis=1)
    avg_iou = best_ious.mean()
    
    # Sort by area (smallest first)
    areas = clusters[:, 0] * clusters[:, 1]
    sorted_indices = np.argsort(areas)
    clusters = clusters[sorted_indices]
    
    return clusters, avg_iou


def assign_anchors_to_scales(
    anchors: np.ndarray,
    num_anchors_16: int,
    num_anchors_8: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign anchors to feature map scales based on size.
    
    Small anchors go to 16x16 (higher resolution, smaller receptive field)
    Large anchors go to 8x8 (lower resolution, larger receptive field)
    
    Args:
        anchors: (K, 2) sorted anchor sizes [w, h]
        num_anchors_16: Number of anchors for 16x16 scale
        num_anchors_8: Number of anchors for 8x8 scale
        
    Returns:
        Tuple of (anchors_16, anchors_8)
    """
    total_anchors = num_anchors_16 + num_anchors_8
    
    if len(anchors) < total_anchors:
        # Pad with default anchors if not enough
        print(f"Warning: Only {len(anchors)} anchors, padding to {total_anchors}")
        padding = np.array([[0.1, 0.1]] * (total_anchors - len(anchors)))
        anchors = np.vstack([anchors, padding])
    elif len(anchors) > total_anchors:
        # Take top-k by coverage
        anchors = anchors[:total_anchors]
    
    # Small anchors for 16x16, large for 8x8
    anchors_16 = anchors[:num_anchors_16]
    anchors_8 = anchors[num_anchors_16:num_anchors_16 + num_anchors_8]
    
    return anchors_16, anchors_8


def analyze_bbox_distribution(wh_data: np.ndarray) -> Dict:
    """
    Analyze bounding box size distribution.
    
    Args:
        wh_data: (N, 2) array of [width, height] pairs
        
    Returns:
        Statistics dictionary
    """
    if len(wh_data) == 0:
        return {}
    
    widths = wh_data[:, 0]
    heights = wh_data[:, 1]
    areas = widths * heights
    aspect_ratios = widths / (heights + 1e-6)
    
    return {
        'count': len(wh_data),
        'width': {
            'min': widths.min(),
            'max': widths.max(),
            'mean': widths.mean(),
            'std': widths.std(),
            'median': np.median(widths),
        },
        'height': {
            'min': heights.min(),
            'max': heights.max(),
            'mean': heights.mean(),
            'std': heights.std(),
            'median': np.median(heights),
        },
        'area': {
            'min': areas.min(),
            'max': areas.max(),
            'mean': areas.mean(),
            'std': areas.std(),
            'median': np.median(areas),
        },
        'aspect_ratio': {
            'min': aspect_ratios.min(),
            'max': aspect_ratios.max(),
            'mean': aspect_ratios.mean(),
            'std': aspect_ratios.std(),
            'median': np.median(aspect_ratios),
        },
    }


def visualize_anchors(
    wh_data: np.ndarray,
    anchors_16: np.ndarray,
    anchors_8: np.ndarray,
    output_path: str,
    stats: Dict,
):
    """
    Create visualization of bbox distribution and anchors.
    
    Args:
        wh_data: (N, 2) width-height pairs
        anchors_16: Anchors for 16x16 scale
        anchors_8: Anchors for 8x8 scale
        output_path: Path to save visualization
        stats: Statistics dictionary
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Width-Height scatter with anchors
    ax1 = axes[0, 0]
    ax1.scatter(wh_data[:, 0], wh_data[:, 1], alpha=0.3, s=10, label='BBoxes')
    ax1.scatter(anchors_16[:, 0], anchors_16[:, 1], c='red', s=200, marker='*', 
                label=f'16x16 anchors ({len(anchors_16)})', edgecolors='black', linewidths=1)
    ax1.scatter(anchors_8[:, 0], anchors_8[:, 1], c='blue', s=200, marker='^',
                label=f'8x8 anchors ({len(anchors_8)})', edgecolors='black', linewidths=1)
    ax1.set_xlabel('Width (normalized)')
    ax1.set_ylabel('Height (normalized)')
    ax1.set_title('BBox Sizes and Anchors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(1.0, wh_data[:, 0].max() * 1.2))
    ax1.set_ylim(0, min(1.0, wh_data[:, 1].max() * 1.2))
    
    # 2. Width histogram
    ax2 = axes[0, 1]
    ax2.hist(wh_data[:, 0], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    for a in anchors_16:
        ax2.axvline(a[0], color='red', linestyle='--', linewidth=2)
    for a in anchors_8:
        ax2.axvline(a[0], color='blue', linestyle='--', linewidth=2)
    ax2.set_xlabel('Width (normalized)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Width Distribution (mean={stats["width"]["mean"]:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Height histogram
    ax3 = axes[1, 0]
    ax3.hist(wh_data[:, 1], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    for a in anchors_16:
        ax3.axvline(a[1], color='red', linestyle='--', linewidth=2)
    for a in anchors_8:
        ax3.axvline(a[1], color='blue', linestyle='--', linewidth=2)
    ax3.set_xlabel('Height (normalized)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Height Distribution (mean={stats["height"]["mean"]:.3f})')
    ax3.grid(True, alpha=0.3)
    
    # 4. Aspect ratio histogram
    ax4 = axes[1, 1]
    aspect_ratios = wh_data[:, 0] / (wh_data[:, 1] + 1e-6)
    ax4.hist(aspect_ratios, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(1.0, color='gray', linestyle='-', linewidth=2, label='Square (1:1)')
    ax4.set_xlabel('Aspect Ratio (W/H)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Aspect Ratio Distribution (mean={stats["aspect_ratio"]["mean"]:.2f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {output_path}")


def generate_anchors(
    train_metadata: str,
    val_metadata: str,
    output_dir: str,
    num_anchors_16: int = 2,
    num_anchors_8: int = 6,
    visualize: bool = True,
) -> Dict:
    """
    Generate optimized anchors from training data.
    
    Args:
        train_metadata: Path to training metadata
        val_metadata: Path to validation metadata
        output_dir: Directory to save outputs
        num_anchors_16: Number of anchors for 16x16 scale
        num_anchors_8: Number of anchors for 8x8 scale
        visualize: Whether to create visualization
        
    Returns:
        Anchor configuration dictionary
    """
    print("=" * 60)
    print("Anchor Generation for Ear Detector")
    print("=" * 60)
    
    # Load bboxes from both train and val
    print(f"\nLoading bboxes from:")
    print(f"  Train: {train_metadata}")
    print(f"  Val: {val_metadata}")
    
    train_bboxes = load_bboxes(train_metadata)
    val_bboxes = load_bboxes(val_metadata)
    
    # Combine all bboxes
    all_bboxes = []
    if len(train_bboxes) > 0:
        all_bboxes.append(train_bboxes)
    if len(val_bboxes) > 0:
        all_bboxes.append(val_bboxes)
    
    if not all_bboxes:
        raise ValueError("No bboxes found in metadata files")
    
    all_bboxes = np.vstack(all_bboxes)
    print(f"\nTotal bboxes: {len(all_bboxes)}")
    
    # Convert to width-height pairs
    # Assuming data is in x,y,w,h format (from preprocessing)
    wh_data = convert_to_wh(all_bboxes, format='xywh')
    
    # Filter out invalid boxes
    valid_mask = (wh_data[:, 0] > 0) & (wh_data[:, 1] > 0)
    wh_data = wh_data[valid_mask]
    print(f"Valid bboxes: {len(wh_data)}")
    
    # Analyze distribution
    stats = analyze_bbox_distribution(wh_data)
    
    print(f"\nBBox Statistics:")
    print(f"  Width:  min={stats['width']['min']:.4f}, max={stats['width']['max']:.4f}, "
          f"mean={stats['width']['mean']:.4f}, median={stats['width']['median']:.4f}")
    print(f"  Height: min={stats['height']['min']:.4f}, max={stats['height']['max']:.4f}, "
          f"mean={stats['height']['mean']:.4f}, median={stats['height']['median']:.4f}")
    print(f"  Aspect: min={stats['aspect_ratio']['min']:.2f}, max={stats['aspect_ratio']['max']:.2f}, "
          f"mean={stats['aspect_ratio']['mean']:.2f}")
    
    # Run K-means clustering
    total_anchors = num_anchors_16 + num_anchors_8
    print(f"\nRunning K-means with {total_anchors} clusters...")
    
    anchors, avg_iou = kmeans_anchors(wh_data, total_anchors)
    print(f"Average IoU with ground truth: {avg_iou:.4f}")
    
    # Assign to scales
    anchors_16, anchors_8 = assign_anchors_to_scales(
        anchors, num_anchors_16, num_anchors_8
    )
    
    print(f"\n16x16 Scale Anchors ({len(anchors_16)}):")
    for i, a in enumerate(anchors_16):
        print(f"  Anchor {i+1}: w={a[0]:.4f}, h={a[1]:.4f}, "
              f"aspect={a[0]/a[1]:.2f}, area={a[0]*a[1]:.6f}")
    
    print(f"\n8x8 Scale Anchors ({len(anchors_8)}):")
    for i, a in enumerate(anchors_8):
        print(f"  Anchor {i+1}: w={a[0]:.4f}, h={a[1]:.4f}, "
              f"aspect={a[0]/a[1]:.2f}, area={a[0]*a[1]:.6f}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save anchor configuration
    anchor_config = {
        'anchors_16': anchors_16,
        'anchors_8': anchors_8,
        'num_anchors_16': num_anchors_16,
        'num_anchors_8': num_anchors_8,
        'avg_iou': avg_iou,
        'stats': stats,
        'total_bboxes': len(wh_data),
    }
    
    output_path = output_dir / 'detector_anchors.npy'
    np.save(output_path, anchor_config, allow_pickle=True)
    print(f"\nSaved anchor config to: {output_path}")
    
    # Create visualization
    if visualize:
        viz_path = output_dir / 'detector_anchor_analysis.png'
        visualize_anchors(wh_data, anchors_16, anchors_8, str(viz_path), stats)
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("Usage in ear_detector:")
    print("=" * 60)
    print("""
# Load anchors in your model:
anchor_config = np.load('data/preprocessed/detector_anchors.npy', allow_pickle=True).item()
anchors_16 = anchor_config['anchors_16']  # Shape: (num_anchors_16, 2) [w, h]
anchors_8 = anchor_config['anchors_8']    # Shape: (num_anchors_8, 2) [w, h]

# Modify model._generate_anchors() to use these values:
# For 16x16: use anchors_16[i] instead of fixed [0.1, 0.1]
# For 8x8: use anchors_8[i] instead of fixed [0.2, 0.2]
""")
    
    return anchor_config


def main():
    parser = argparse.ArgumentParser(
        description='Generate optimized anchors for ear detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with defaults
    python create_detector_anchors.py
    
    # Custom anchor counts
    python create_detector_anchors.py --num_anchors_16 3 --num_anchors_8 6
    
    # Custom paths
    python create_detector_anchors.py \\
        --train_metadata data/preprocessed/train_detector.npy \\
        --output_dir data/preprocessed
        """
    )
    
    parser.add_argument(
        '--train_metadata',
        type=str,
        default='data/preprocessed/train_detector.npy',
        help='Path to training metadata file',
    )
    parser.add_argument(
        '--val_metadata',
        type=str,
        default='data/preprocessed/val_detector.npy',
        help='Path to validation metadata file',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/preprocessed',
        help='Output directory for anchor files',
    )
    parser.add_argument(
        '--num_anchors_16',
        type=int,
        default=2,
        help='Number of anchors for 16x16 feature map (default: 2)',
    )
    parser.add_argument(
        '--num_anchors_8',
        type=int,
        default=6,
        help='Number of anchors for 8x8 feature map (default: 6)',
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Disable visualization generation',
    )
    
    args = parser.parse_args()
    
    generate_anchors(
        train_metadata=args.train_metadata,
        val_metadata=args.val_metadata,
        output_dir=args.output_dir,
        num_anchors_16=args.num_anchors_16,
        num_anchors_8=args.num_anchors_8,
        visualize=not args.no_visualize,
    )


if __name__ == '__main__':
    main()
