"""
Training Analysis Script - Comprehensive analysis of training runs.

Analyzes PyTorch Lightning training logs and provides insights on:
- Training progress and convergence
- Overfitting detection
- Learning rate analysis
- Performance metrics
- Recommendations for improvement

Usage:
    python common/training_analysis.py ear_detector
    python common/training_analysis.py ear_detector --version 0
    python common/training_analysis.py ear_landmarker --output analysis.png
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import yaml

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrainingStatus(Enum):
    """Training run status."""
    HEALTHY = "healthy"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    UNSTABLE = "unstable"
    EARLY_STOPPED = "early_stopped"
    DIVERGED = "diverged"


@dataclass
class TrainingMetrics:
    """Container for training metrics analysis."""
    # Basic info
    total_epochs: int
    total_steps: int
    
    # Loss metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_val_loss_epoch: int
    
    # mAP metrics (if available)
    final_map: Optional[float] = None
    final_map_50: Optional[float] = None
    final_map_75: Optional[float] = None
    best_map_50: Optional[float] = None
    best_map_50_epoch: Optional[int] = None
    
    # Learning rate
    final_lr: Optional[float] = None
    max_lr: Optional[float] = None
    
    # FPS (if available)
    avg_fps: Optional[float] = None
    
    # Convergence analysis
    train_loss_trend: float = 0.0  # Slope of last 20% of training
    val_loss_trend: float = 0.0
    loss_gap: float = 0.0  # train_loss - val_loss (negative = overfitting)
    
    # Status
    status: TrainingStatus = TrainingStatus.HEALTHY


def load_metrics_csv(csv_path: Path) -> pd.DataFrame:
    """Load and clean metrics CSV file."""
    df = pd.read_csv(csv_path)
    
    # Forward fill epoch numbers
    df['epoch'] = df['epoch'].ffill()
    
    return df


def load_hparams(yaml_path: Path) -> Dict[str, Any]:
    """Load hyperparameters from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-epoch validation metrics."""
    # Get rows with validation metrics
    val_cols = [c for c in df.columns if c.startswith('val/')]
    
    if not val_cols:
        return pd.DataFrame()
    
    # Filter rows that have validation data
    val_df = df[df[val_cols[0]].notna()].copy()
    
    # Get unique epochs
    epoch_metrics = val_df.groupby('epoch').last().reset_index()
    
    return epoch_metrics


def get_train_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract per-epoch training metrics."""
    train_epoch_cols = [c for c in df.columns if c.endswith('_epoch') and 'train' in c]
    
    if not train_epoch_cols:
        return pd.DataFrame()
    
    # Filter rows with epoch-level training data
    train_df = df[df[train_epoch_cols[0]].notna()].copy()
    
    # Get unique epochs
    epoch_metrics = train_df.groupby('epoch').last().reset_index()
    
    return epoch_metrics


def analyze_training(
    metrics_df: pd.DataFrame,
    hparams: Dict[str, Any],
) -> TrainingMetrics:
    """Perform comprehensive training analysis."""
    
    # Get epoch-level metrics
    val_epochs = get_epoch_metrics(metrics_df)
    train_epochs = get_train_metrics(metrics_df)
    
    # Basic counts
    total_epochs = int(metrics_df['epoch'].max()) + 1 if not metrics_df['epoch'].isna().all() else 0
    total_steps = int(metrics_df['step'].max()) if 'step' in metrics_df else 0
    
    # Initialize metrics
    metrics = TrainingMetrics(
        total_epochs=total_epochs,
        total_steps=total_steps,
        final_train_loss=0.0,
        final_val_loss=0.0,
        best_val_loss=float('inf'),
        best_val_loss_epoch=0,
    )
    
    # Training loss analysis
    if len(train_epochs) > 0 and 'train/loss_epoch' in train_epochs.columns:
        train_loss = train_epochs['train/loss_epoch'].dropna()
        if len(train_loss) > 0:
            metrics.final_train_loss = float(train_loss.iloc[-1])
            
            # Trend analysis (last 20% of training)
            window = max(1, len(train_loss) // 5)
            if len(train_loss) > window:
                recent = train_loss.iloc[-window:].values
                x = np.arange(len(recent))
                # Check for valid data before polyfit
                if len(recent) >= 2 and not np.any(np.isnan(recent)) and np.std(recent) > 1e-10:
                    try:
                        slope, _ = np.polyfit(x, recent, 1)
                        metrics.train_loss_trend = float(slope)
                    except (np.linalg.LinAlgError, ValueError):
                        metrics.train_loss_trend = 0.0
    
    # Validation loss analysis
    if len(val_epochs) > 0 and 'val/loss' in val_epochs.columns:
        val_loss = val_epochs['val/loss'].dropna()
        if len(val_loss) > 0:
            metrics.final_val_loss = float(val_loss.iloc[-1])
            metrics.best_val_loss = float(val_loss.min())
            metrics.best_val_loss_epoch = int(val_epochs.loc[val_loss.idxmin(), 'epoch'])
            
            # Trend analysis
            window = max(1, len(val_loss) // 5)
            if len(val_loss) > window:
                recent = val_loss.iloc[-window:].values
                x = np.arange(len(recent))
                # Check for valid data before polyfit
                if len(recent) >= 2 and not np.any(np.isnan(recent)) and np.std(recent) > 1e-10:
                    try:
                        slope, _ = np.polyfit(x, recent, 1)
                        metrics.val_loss_trend = float(slope)
                    except (np.linalg.LinAlgError, ValueError):
                        metrics.val_loss_trend = 0.0
    
    # Loss gap (overfitting indicator)
    if metrics.final_train_loss > 0 and metrics.final_val_loss > 0:
        metrics.loss_gap = metrics.final_val_loss - metrics.final_train_loss
    
    # mAP metrics
    if len(val_epochs) > 0:
        if 'val/mAP' in val_epochs.columns:
            map_vals = val_epochs['val/mAP'].dropna()
            if len(map_vals) > 0:
                metrics.final_map = float(map_vals.iloc[-1])
        
        if 'val/mAP_50' in val_epochs.columns:
            map50_vals = val_epochs['val/mAP_50'].dropna()
            if len(map50_vals) > 0:
                metrics.final_map_50 = float(map50_vals.iloc[-1])
                metrics.best_map_50 = float(map50_vals.max())
                metrics.best_map_50_epoch = int(val_epochs.loc[map50_vals.idxmax(), 'epoch'])
        
        if 'val/mAP_75' in val_epochs.columns:
            map75_vals = val_epochs['val/mAP_75'].dropna()
            if len(map75_vals) > 0:
                metrics.final_map_75 = float(map75_vals.iloc[-1])
    
    # Learning rate
    if 'train/lr' in metrics_df.columns:
        lr_vals = metrics_df['train/lr'].dropna()
        if len(lr_vals) > 0:
            metrics.final_lr = float(lr_vals.iloc[-1])
            metrics.max_lr = float(lr_vals.max())
    
    # FPS
    if len(val_epochs) > 0 and 'val/fps' in val_epochs.columns:
        fps_vals = val_epochs['val/fps'].dropna()
        if len(fps_vals) > 0:
            metrics.avg_fps = float(fps_vals.mean())
    
    # Determine training status
    metrics.status = determine_status(metrics, hparams)
    
    return metrics


def determine_status(metrics: TrainingMetrics, hparams: Dict) -> TrainingStatus:
    """Determine the overall training status."""
    max_epochs = hparams.get('max_epochs', 100)
    
    # Check for divergence
    if metrics.final_val_loss > 10 or np.isnan(metrics.final_val_loss):
        return TrainingStatus.DIVERGED
    
    # Check for early stopping (didn't complete all epochs)
    if metrics.total_epochs < max_epochs * 0.9:
        return TrainingStatus.EARLY_STOPPED
    
    # Check for overfitting (val loss much higher than train, increasing trend)
    if metrics.loss_gap > 0.1 * metrics.final_train_loss and metrics.val_loss_trend > 0:
        return TrainingStatus.OVERFITTING
    
    # Check for underfitting (both losses still high and decreasing)
    if metrics.train_loss_trend < -0.001 and metrics.val_loss_trend < -0.001:
        return TrainingStatus.UNDERFITTING
    
    # Check for unstable training (high variance)
    if abs(metrics.train_loss_trend) > 0.01:
        return TrainingStatus.UNSTABLE
    
    return TrainingStatus.HEALTHY


def generate_recommendations(
    metrics: TrainingMetrics,
    hparams: Dict[str, Any],
) -> List[str]:
    """Generate improvement recommendations based on analysis."""
    recommendations = []
    
    # Status-based recommendations
    if metrics.status == TrainingStatus.OVERFITTING:
        recommendations.extend([
            "🔴 OVERFITTING DETECTED:",
            "  - Increase dropout or add regularization",
            "  - Add more data augmentation",
            "  - Reduce model capacity or add early stopping",
            "  - Increase weight_decay (currently: {})".format(hparams.get('weight_decay', 'N/A')),
        ])
    
    elif metrics.status == TrainingStatus.UNDERFITTING:
        recommendations.extend([
            "🟡 UNDERFITTING - Model still improving:",
            "  - Train for more epochs (currently: {})".format(hparams.get('max_epochs', 'N/A')),
            "  - Increase model capacity",
            "  - Increase learning rate",
        ])
    
    elif metrics.status == TrainingStatus.UNSTABLE:
        recommendations.extend([
            "🟠 UNSTABLE TRAINING:",
            "  - Reduce learning rate (currently: {})".format(hparams.get('learning_rate', 'N/A')),
            "  - Increase batch size",
            "  - Add gradient clipping",
            "  - Check for data quality issues",
        ])
    
    elif metrics.status == TrainingStatus.DIVERGED:
        recommendations.extend([
            "🔴 TRAINING DIVERGED:",
            "  - Significantly reduce learning rate",
            "  - Check loss function implementation",
            "  - Verify data normalization",
            "  - Check for NaN values in data",
        ])
    
    # mAP-based recommendations
    if metrics.final_map_50 is not None:
        if metrics.final_map_50 < 0.3:
            recommendations.extend([
                "",
                "📊 LOW mAP@0.5 ({:.4f}):".format(metrics.final_map_50),
                "  - Consider using custom anchors (run create_detector_anchors.py)",
                "  - Review anchor IoU thresholds",
                "  - Check bounding box preprocessing",
                "  - Increase training data or augmentation",
            ])
        elif metrics.final_map_50 < 0.6:
            recommendations.extend([
                "",
                "📊 MODERATE mAP@0.5 ({:.4f}):".format(metrics.final_map_50),
                "  - Fine-tune anchor sizes for your dataset",
                "  - Try different IoU thresholds for NMS",
                "  - Consider multi-scale training",
            ])
    
    # Learning rate recommendations
    if metrics.final_lr is not None and metrics.max_lr is not None:
        if metrics.final_lr < metrics.max_lr * 0.01:
            recommendations.extend([
                "",
                "📉 Learning rate decayed significantly:",
                "  - Consider cosine annealing with restarts",
                "  - Try OneCycleLR for faster convergence",
            ])
    
    # FPS recommendations
    if metrics.avg_fps is not None:
        if metrics.avg_fps < 30:
            recommendations.extend([
                "",
                "⚡ LOW INFERENCE SPEED ({:.1f} FPS):".format(metrics.avg_fps),
                "  - Consider model pruning or quantization",
                "  - Reduce input resolution if acceptable",
            ])
    
    # Best epoch analysis
    if metrics.best_map_50_epoch is not None:
        epochs_since_best = metrics.total_epochs - metrics.best_map_50_epoch - 1
        if epochs_since_best > 20:
            recommendations.extend([
                "",
                "⏰ BEST MODEL WAS {} EPOCHS AGO:".format(epochs_since_best),
                "  - Consider early stopping",
                "  - Use the checkpoint from epoch {}".format(metrics.best_map_50_epoch),
            ])
    
    # General improvements
    if not recommendations:
        recommendations = [
            "✅ Training looks healthy!",
            "",
            "📈 Potential improvements:",
            "  - Try data-driven anchors (create_detector_anchors.py)",
            "  - Experiment with different augmentations",
            "  - Try GIoU/DIoU loss instead of Smooth L1",
            "  - Add Exponential Moving Average (EMA)",
        ]
    
    return recommendations


def print_analysis(
    metrics: TrainingMetrics,
    hparams: Dict[str, Any],
    recommendations: List[str],
    output_dir: Path,
):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 70)
    print("TRAINING ANALYSIS REPORT")
    print("=" * 70)
    
    # Training overview
    print("\n📊 TRAINING OVERVIEW")
    print("-" * 40)
    print(f"  Total Epochs:     {metrics.total_epochs}")
    print(f"  Total Steps:      {metrics.total_steps}")
    print(f"  Status:           {metrics.status.value.upper()}")
    
    # Loss metrics
    print("\n📉 LOSS METRICS")
    print("-" * 40)
    print(f"  Final Train Loss: {metrics.final_train_loss:.6f}")
    print(f"  Final Val Loss:   {metrics.final_val_loss:.6f}")
    print(f"  Best Val Loss:    {metrics.best_val_loss:.6f} (epoch {metrics.best_val_loss_epoch})")
    print(f"  Loss Gap:         {metrics.loss_gap:+.6f}")
    print(f"  Train Trend:      {metrics.train_loss_trend:+.6f} (slope)")
    print(f"  Val Trend:        {metrics.val_loss_trend:+.6f} (slope)")
    
    # Detection metrics
    if metrics.final_map is not None or metrics.final_map_50 is not None:
        print("\n🎯 DETECTION METRICS")
        print("-" * 40)
        if metrics.final_map is not None:
            print(f"  Final mAP:        {metrics.final_map:.4f}")
        if metrics.final_map_50 is not None:
            print(f"  Final mAP@0.5:    {metrics.final_map_50:.4f}")
            print(f"  Best mAP@0.5:     {metrics.best_map_50:.4f} (epoch {metrics.best_map_50_epoch})")
        if metrics.final_map_75 is not None:
            print(f"  Final mAP@0.75:   {metrics.final_map_75:.4f}")
    
    # Performance
    if metrics.avg_fps is not None:
        print("\n⚡ PERFORMANCE")
        print("-" * 40)
        print(f"  Avg FPS:          {metrics.avg_fps:.1f}")
    
    # Learning rate
    if metrics.final_lr is not None:
        print("\n📈 LEARNING RATE")
        print("-" * 40)
        print(f"  Final LR:         {metrics.final_lr:.2e}")
        print(f"  Max LR:           {metrics.max_lr:.2e}")
    
    # Hyperparameters summary
    print("\n⚙️ HYPERPARAMETERS")
    print("-" * 40)
    key_hparams = ['learning_rate', 'weight_decay', 'batch_size', 'max_epochs', 
                   'warmup_epochs', 'focal_alpha', 'focal_gamma', 'box_weight']
    for key in key_hparams:
        if key in hparams:
            print(f"  {key}: {hparams[key]}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 70)


def plot_training_curves(
    metrics_df: pd.DataFrame,
    output_path: Path,
):
    """Generate training curves visualization."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping visualization")
        return
    
    val_epochs = get_epoch_metrics(metrics_df)
    train_epochs = get_train_metrics(metrics_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    if len(train_epochs) > 0 and 'train/loss_epoch' in train_epochs.columns:
        ax1.plot(train_epochs['epoch'], train_epochs['train/loss_epoch'], 
                 label='Train Loss', color='blue', alpha=0.8)
    if len(val_epochs) > 0 and 'val/loss' in val_epochs.columns:
        ax1.plot(val_epochs['epoch'], val_epochs['val/loss'], 
                 label='Val Loss', color='orange', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. mAP curves
    ax2 = axes[0, 1]
    if len(val_epochs) > 0:
        if 'val/mAP_50' in val_epochs.columns:
            map50 = val_epochs['val/mAP_50'].dropna()
            ax2.plot(val_epochs.loc[map50.index, 'epoch'], map50, 
                     label='mAP@0.5', color='green', linewidth=2)
        if 'val/mAP_75' in val_epochs.columns:
            map75 = val_epochs['val/mAP_75'].dropna()
            ax2.plot(val_epochs.loc[map75.index, 'epoch'], map75, 
                     label='mAP@0.75', color='purple', linewidth=2)
        if 'val/mAP' in val_epochs.columns:
            map_all = val_epochs['val/mAP'].dropna()
            ax2.plot(val_epochs.loc[map_all.index, 'epoch'], map_all, 
                     label='mAP', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP')
    ax2.set_title('Detection Metrics (mAP)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Learning rate
    ax3 = axes[1, 0]
    if 'train/lr' in metrics_df.columns:
        lr_df = metrics_df[['step', 'train/lr']].dropna()
        if len(lr_df) > 0:
            ax3.plot(lr_df['step'], lr_df['train/lr'], color='teal', alpha=0.8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # 4. Component losses
    ax4 = axes[1, 1]
    if len(val_epochs) > 0:
        if 'val/cls_loss' in val_epochs.columns:
            cls_loss = val_epochs['val/cls_loss'].dropna()
            ax4.plot(val_epochs.loc[cls_loss.index, 'epoch'], cls_loss, 
                     label='Classification Loss', color='coral')
        if 'val/box_loss' in val_epochs.columns:
            box_loss = val_epochs['val/box_loss'].dropna()
            ax4.plot(val_epochs.loc[box_loss.index, 'epoch'], box_loss, 
                     label='Box Regression Loss', color='steelblue')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Component Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved training curves to: {output_path}")


def find_latest_version(logs_dir: Path) -> Optional[Path]:
    """Find the latest version directory."""
    if not logs_dir.exists():
        return None
    
    versions = []
    for d in logs_dir.iterdir():
        if d.is_dir() and d.name.startswith('version_'):
            try:
                version_num = int(d.name.split('_')[1])
                versions.append((version_num, d))
            except (ValueError, IndexError):
                continue
    
    if not versions:
        return None
    
    # Return highest version number
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training run and provide recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python common/training_analysis.py ear_detector
    python common/training_analysis.py ear_detector --version 0
    python common/training_analysis.py ear_landmarker --output curves.png
        """
    )
    
    parser.add_argument(
        'module',
        type=str,
        help='Module name (e.g., ear_detector, ear_landmarker)',
    )
    parser.add_argument(
        '--version',
        type=int,
        default=None,
        help='Specific version number to analyze (default: latest)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for training curves plot',
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Skip generating plots',
    )
    
    args = parser.parse_args()
    
    # Find module directory
    script_dir = Path(__file__).parent.parent
    module_dir = script_dir / args.module
    
    if not module_dir.exists():
        print(f"Error: Module directory not found: {module_dir}")
        sys.exit(1)
    
    outputs_dir = module_dir / 'outputs'
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        sys.exit(1)
    
    # Find logs directory (usually named after the experiment)
    logs_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name != 'checkpoints']
    
    if not logs_dirs:
        print(f"Error: No log directories found in {outputs_dir}")
        sys.exit(1)
    
    # Use the first logs directory (usually there's only one)
    logs_dir = logs_dirs[0]
    
    # Find version directory
    if args.version is not None:
        version_dir = logs_dir / f'version_{args.version}'
        if not version_dir.exists():
            print(f"Error: Version directory not found: {version_dir}")
            sys.exit(1)
    else:
        version_dir = find_latest_version(logs_dir)
        if version_dir is None:
            print(f"Error: No version directories found in {logs_dir}")
            sys.exit(1)
    
    print(f"Analyzing: {version_dir}")
    
    # Load data
    metrics_path = version_dir / 'metrics.csv'
    hparams_path = version_dir / 'hparams.yaml'
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    metrics_df = load_metrics_csv(metrics_path)
    hparams = load_hparams(hparams_path) if hparams_path.exists() else {}
    
    # Analyze
    metrics = analyze_training(metrics_df, hparams)
    recommendations = generate_recommendations(metrics, hparams)
    
    # Print report
    print_analysis(metrics, hparams, recommendations, version_dir)
    
    # Generate plots
    if not args.no_plot:
        output_path = args.output or (version_dir / 'training_analysis.png')
        plot_training_curves(metrics_df, Path(output_path))


if __name__ == '__main__':
    main()
