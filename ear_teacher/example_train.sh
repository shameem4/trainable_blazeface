#!/bin/bash
# Example training script for Ear Teacher VAE

# Basic training with default settings
python train.py

# Advanced training with custom hyperparameters
# python train.py \
#   --train-npy data/preprocessed/train_teacher.npy \
#   --val-npy data/preprocessed/val_teacher.npy \
#   --batch-size 64 \
#   --epochs 300 \
#   --lr 2e-4 \
#   --latent-dim 512 \
#   --image-size 128 \
#   --kl-weight 0.0001 \
#   --perceptual-weight 0.5 \
#   --ssim-weight 0.1 \
#   --recon-loss mse \
#   --warmup-epochs 5 \
#   --scheduler cosine \
#   --num-workers 8 \
#   --gpus 1 \
#   --precision 16-mixed \
#   --early-stopping \
#   --patience 30 \
#   --save-dir checkpoints/ear_teacher \
#   --log-dir logs/ear_teacher
