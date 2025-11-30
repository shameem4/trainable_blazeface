"""Visualize KLD annealing schedule with warmup."""

import matplotlib.pyplot as plt
import numpy as np
import math


def cyclic_annealing(epoch, max_epochs, warmup_epochs, n_cycles, ratio, start, end, kld_weight):
    """Calculate KLD weight for cyclic annealing with warmup."""
    # Warmup period
    if epoch < warmup_epochs:
        return 0.0

    # Adjust for warmup
    total_epochs = max_epochs - warmup_epochs
    current_epoch_adjusted = epoch - warmup_epochs

    # Calculate cycle parameters
    cycle_length = total_epochs / n_cycles
    current_position = current_epoch_adjusted % cycle_length

    # Within each cycle
    increase_length = cycle_length * ratio

    if current_position < increase_length:
        # Increasing phase
        progress = current_position / increase_length
    else:
        # Constant phase at maximum
        progress = 1.0

    # Apply start and end scaling
    current_weight = start + (end - start) * progress

    return current_weight * kld_weight


# Parameters matching your current setup
max_epochs = 200
warmup_epochs = 10
n_cycles = 4
ratio = 0.5
start = 0.0
end = 1.0
kld_weight = 0.00025

# Calculate KLD weights for each epoch
epochs = np.arange(max_epochs)
kld_weights = [cyclic_annealing(e, max_epochs, warmup_epochs, n_cycles, ratio, start, end, kld_weight)
               for e in epochs]

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Full schedule
ax1.plot(epochs, kld_weights, linewidth=2, color='#2E86AB')
ax1.axvline(x=warmup_epochs, color='red', linestyle='--', linewidth=2, label=f'Warmup End (Epoch {warmup_epochs})')
ax1.axhspan(0, max(kld_weights), xmin=0, xmax=warmup_epochs/max_epochs, alpha=0.1, color='red', label='Warmup Period')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('KLD Weight', fontsize=12)
ax1.set_title(f'Cyclic KLD Annealing Schedule with {warmup_epochs}-Epoch Warmup\n'
              f'({n_cycles} cycles, ratio={ratio}, max_weight={kld_weight})',
              fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xlim(0, max_epochs)

# Zoomed view of first 50 epochs
ax2.plot(epochs[:50], kld_weights[:50], linewidth=2, color='#2E86AB')
ax2.axvline(x=warmup_epochs, color='red', linestyle='--', linewidth=2, label=f'Warmup End (Epoch {warmup_epochs})')
ax2.axhspan(0, max(kld_weights[:50]), xmin=0, xmax=warmup_epochs/50, alpha=0.1, color='red', label='Warmup Period')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('KLD Weight', fontsize=12)
ax2.set_title('First 50 Epochs (Detailed View)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim(0, 50)

plt.tight_layout()
plt.savefig('ear_teacher/kld_annealing_schedule.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to: ear_teacher/kld_annealing_schedule.png")

# Print some key values
print(f"\nKLD Annealing Schedule:")
print(f"  Total epochs: {max_epochs}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  Number of cycles: {n_cycles}")
print(f"  Cycle ratio: {ratio}")
print(f"  Max KLD weight: {kld_weight}")
print(f"\nKey epoch values:")
print(f"  Epoch 0-{warmup_epochs-1}: {kld_weights[0]:.6f} (warmup)")
print(f"  Epoch {warmup_epochs}: {kld_weights[warmup_epochs]:.6f} (annealing starts)")
print(f"  Epoch 20: {kld_weights[20]:.6f}")
print(f"  Epoch 50: {kld_weights[50]:.6f}")
print(f"  Epoch 100: {kld_weights[100]:.6f}")
print(f"  Epoch 199: {kld_weights[199]:.6f} (final)")
