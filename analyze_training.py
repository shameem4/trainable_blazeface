"""Analyze training metrics and suggest hyperparameter adjustments."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read metrics
metrics_file = Path('ear_teacher/logs/ear_vae/version_0/metrics.csv')
df = pd.read_csv(metrics_file)

# Filter to validation metrics only
val_df = df[df['val/loss'].notna()].copy()

print("="*80)
print("TRAINING ANALYSIS - Last Run")
print("="*80)
print(f"\nTotal epochs trained: {len(val_df)}")
print(f"Training stopped at epoch: {val_df['epoch'].max()}")

# Latest metrics
latest = val_df.iloc[-1]
print(f"\n{'LATEST VALIDATION METRICS (Epoch ' + str(int(latest['epoch'])) + ')':^80}")
print("="*80)
print(f"  Val Loss:        {latest['val/loss']:.6f}")
print(f"  Recon Loss:      {latest['val/recon_loss']:.6f}")
print(f"  KLD:             {latest['val/kld']:.2f}")
print(f"  Perceptual:      {latest['val/perceptual_loss']:.4f}")
print(f"  SSIM Loss:       {latest['val/ssim_loss']:.4f}")
print(f"  SSIM (metric):   {latest['val/ssim']:.4f} (target: >0.7)")
print(f"  Edge Loss:       {latest['val/edge_loss']:.4f}")
print(f"  PSNR:            {latest['val/psnr']:.2f} dB (target: >25 dB)")

# Training metrics
train_latest = val_df.iloc[-1]
print(f"\n{'LATEST TRAINING METRICS':^80}")
print("="*80)
print(f"  Train Loss:      {train_latest['train/loss_epoch']:.6f}")
print(f"  Recon Loss:      {train_latest['train/recon_loss']:.6f}")
print(f"  KLD:             {train_latest['train/kld']:.2f}")
print(f"  KLD Weight:      {train_latest['train/kld_weight']:.8f}")
print(f"  Perceptual:      {train_latest['train/perceptual_loss']:.4f}")
print(f"  SSIM (metric):   {train_latest['train/ssim']:.4f}")
print(f"  PSNR:            {train_latest['train/psnr']:.2f} dB")

# Progress analysis
first_5 = val_df.iloc[:5]
last_5 = val_df.iloc[-5:]

print(f"\n{'PROGRESS ANALYSIS':^80}")
print("="*80)
print(f"  {'Metric':<20} {'First 5 Avg':<15} {'Last 5 Avg':<15} {'Change':<15}")
print("  " + "-"*76)
change_loss = ((last_5['val/loss'].mean() - first_5['val/loss'].mean()) / first_5['val/loss'].mean() * 100)
change_ssim = ((last_5['val/ssim'].mean() - first_5['val/ssim'].mean()) / first_5['val/ssim'].mean() * 100)
change_psnr = ((last_5['val/psnr'].mean() - first_5['val/psnr'].mean()) / first_5['val/psnr'].mean() * 100)
change_recon = ((last_5['val/recon_loss'].mean() - first_5['val/recon_loss'].mean()) / first_5['val/recon_loss'].mean() * 100)

print(f"  {'Val Loss':<20} {first_5['val/loss'].mean():<15.6f} {last_5['val/loss'].mean():<15.6f} {change_loss:>+14.2f}%")
print(f"  {'SSIM':<20} {first_5['val/ssim'].mean():<15.4f} {last_5['val/ssim'].mean():<15.4f} {change_ssim:>+14.2f}%")
print(f"  {'PSNR':<20} {first_5['val/psnr'].mean():<15.2f} {last_5['val/psnr'].mean():<15.2f} {change_psnr:>+14.2f}%")
print(f"  {'Recon Loss':<20} {first_5['val/recon_loss'].mean():<15.6f} {last_5['val/recon_loss'].mean():<15.6f} {change_recon:>+14.2f}%")

# Loss composition analysis
print(f"\n{'LOSS COMPOSITION (Latest Epoch)':^80}")
print("="*80)

recon_contrib = 1.25 * latest['val/recon_loss']  # recon_weight = 1.25
kld_contrib = 0.00025 * latest['val/kld']  # kld_weight = 0.00025
perc_contrib = 0.005 * latest['val/perceptual_loss']  # perceptual_weight = 0.005
ssim_contrib = 0.05 * latest['val/ssim_loss']  # ssim_weight = 0.05
edge_contrib = 0.05 * latest['val/edge_loss']  # edge_weight = 0.05

total = recon_contrib + kld_contrib + perc_contrib + ssim_contrib + edge_contrib

print(f"  {'Loss Component':<25} {'Raw Value':<15} {'Weighted':<15} {'% of Total':<15}")
print("  " + "-"*76)
print(f"  {'Reconstruction (1.25x)':<25} {latest['val/recon_loss']:<15.6f} {recon_contrib:<15.6f} {recon_contrib/total*100:<14.2f}%")
print(f"  {'KLD (0.00025x)':<25} {latest['val/kld']:<15.2f} {kld_contrib:<15.6f} {kld_contrib/total*100:<14.2f}%")
print(f"  {'Perceptual (0.005x)':<25} {latest['val/perceptual_loss']:<15.4f} {perc_contrib:<15.6f} {perc_contrib/total*100:<14.2f}%")
print(f"  {'SSIM (0.05x)':<25} {latest['val/ssim_loss']:<15.4f} {ssim_contrib:<15.6f} {ssim_contrib/total*100:<14.2f}%")
print(f"  {'Edge (0.05x)':<25} {latest['val/edge_loss']:<15.4f} {edge_contrib:<15.6f} {edge_contrib/total*100:<14.2f}%")
print("  " + "-"*76)
print(f"  {'TOTAL':<25} {'':<15} {total:<15.6f} {'100.00%':<15}")

# Check for convergence issues
print(f"\n{'CONVERGENCE ANALYSIS':^80}")
print("="*80)

# Check if loss is still decreasing
recent_10 = val_df.tail(10)
loss_trend = np.polyfit(range(len(recent_10)), recent_10['val/loss'], 1)[0]

if abs(loss_trend) < 0.0001:
    print("  ⚠ WARNING: Loss has plateaued (not decreasing)")
    print(f"    Loss trend (last 10 epochs): {loss_trend:.8f}")
elif loss_trend > 0:
    print("  ⚠ WARNING: Loss is increasing!")
    print(f"    Loss trend (last 10 epochs): {loss_trend:.8f}")
else:
    print(f"  ✓ Loss is still decreasing (trend: {loss_trend:.8f})")

# Check SSIM improvement
ssim_trend = np.polyfit(range(len(recent_10)), recent_10['val/ssim'], 1)[0]
if ssim_trend < 0.0001:
    print(f"  ⚠ WARNING: SSIM has plateaued")
    print(f"    SSIM trend (last 10 epochs): {ssim_trend:.8f}")
else:
    print(f"  ✓ SSIM is improving (trend: {ssim_trend:.8f})")

# RECOMMENDATIONS
print(f"\n{'RECOMMENDATIONS FOR HYPERPARAMETER TUNING':^80}")
print("="*80)

recommendations = []

# Check SSIM
if latest['val/ssim'] < 0.6:
    recommendations.append(("CRITICAL", "SSIM is very low (<0.6)", [
        "Increase --recon_weight from 1.25 to 2.0-3.0",
        "Reduce --perceptual_weight from 0.005 to 0.001-0.002",
        "Increase --ssim_weight from 0.05 to 0.1-0.15"
    ]))
elif latest['val/ssim'] < 0.7:
    recommendations.append(("WARNING", "SSIM is below target (<0.7)", [
        "Increase --recon_weight from 1.25 to 1.5-2.0",
        "Increase --ssim_weight from 0.05 to 0.08-0.1"
    ]))

# Check PSNR
if latest['val/psnr'] < 20:
    recommendations.append(("CRITICAL", "PSNR is very low (<20 dB)", [
        "Increase --recon_weight significantly (2.0-3.0)",
        "Consider reducing auxiliary losses"
    ]))
elif latest['val/psnr'] < 25:
    recommendations.append(("WARNING", "PSNR is below target (<25 dB)", [
        "Increase --recon_weight to 1.5-2.0"
    ]))

# Check KLD
if latest['val/kld'] < 50:
    recommendations.append(("INFO", "KLD is very low - latent space may be underutilized", [
        "KLD weight will increase after warmup period",
        "Current warmup_epochs: 10, wait for annealing to kick in"
    ]))
elif latest['val/kld'] > 2000:
    recommendations.append(("WARNING", "KLD is very high - possible posterior collapse", [
        "Reduce --kld_weight from 0.00025 to 0.0001",
        "Increase --kld_warmup_epochs from 10 to 15-20"
    ]))

# Check perceptual loss dominance
if perc_contrib / total > 0.4:
    recommendations.append(("WARNING", "Perceptual loss is dominating (>40%)", [
        "Reduce --perceptual_weight from 0.005 to 0.002-0.003"
    ]))

# Check convergence
if abs(loss_trend) < 0.0001 and latest['val/ssim'] < 0.7:
    recommendations.append(("WARNING", "Loss plateaued but quality still low", [
        "Consider reducing learning rate: --learning_rate 5e-5",
        "Or increase reconstruction weight for better pixel fidelity"
    ]))

if len(recommendations) == 0:
    print("  ✓ Training appears to be progressing well!")
    print("  Consider monitoring for a few more epochs")
else:
    for level, issue, suggestions in recommendations:
        print(f"\n  [{level}] {issue}")
        for i, sug in enumerate(suggestions, 1):
            print(f"    {i}. {sug}")

# Suggested command
print(f"\n{'SUGGESTED TRAINING COMMAND':^80}")
print("="*80)

# Build recommended hyperparameters
suggested_params = []
if latest['val/ssim'] < 0.6:
    suggested_params.extend([
        "--recon_weight 2.5",
        "--perceptual_weight 0.002",
        "--ssim_weight 0.12",
        "--edge_weight 0.05",
        "--kld_weight 0.00025",
        "--kld_warmup_epochs 15"
    ])
elif latest['val/ssim'] < 0.7:
    suggested_params.extend([
        "--recon_weight 1.75",
        "--perceptual_weight 0.003",
        "--ssim_weight 0.08",
        "--edge_weight 0.05",
        "--kld_weight 0.00025",
        "--kld_warmup_epochs 12"
    ])
else:
    suggested_params.extend([
        "--recon_weight 1.5",
        "--perceptual_weight 0.004",
        "--ssim_weight 0.06",
        "--edge_weight 0.05"
    ])

print(f"\npython ear_teacher/train.py \\")
for param in suggested_params:
    print(f"  {param} \\")
print(f"  --epochs 200 \\")
print(f"  --resume")

print("\n" + "="*80)
