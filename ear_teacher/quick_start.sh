#!/bin/bash
# Quick start script for ear teacher training
# Run from root directory: bash ear_teacher/quick_start.sh

echo "Starting Ear Teacher Training..."
echo "================================"
echo ""

python -m ear_teacher.train \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --warmup_epochs 10 \
    --precision 16-mixed \
    --devices 1 \
    --num_workers 4 \
    --experiment_name baseline_no_aug \
    --output_dir outputs/ear_teacher

echo ""
echo "Training complete!"
echo "View results with: tensorboard --logdir outputs/ear_teacher"
