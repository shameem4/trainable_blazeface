# Data Processing Usage Guide

## Overview

The data processor now supports:
- **Background disk writing**: Processing continues while data is flushed to disk
- **Lazy loading**: Load NPZ data on-demand during training instead of loading everything into memory

## Processing Data

### Basic Usage

```python
from shared.data_processing.data_processor import DataProcessor

# Create processor
processor = DataProcessor(
    raw_data_dir='data/raw',
    output_dir='data/preprocessed',
    batch_size=500,
    num_workers=8,
    flush_every=5  # Flush to disk every 5 batches
)

# Process all datasets
processor.process_all(train_split=0.8, include_teacher=True)
```

### Command Line

```bash
# Process all datasets
python -m shared.data_processing.data_processor --all

# Process specific datasets
python -m shared.data_processing.data_processor --detector --landmarker

# Custom settings
python -m shared.data_processing.data_processor --all --batch-size 1000 --workers 8 --flush-every 10
```

### Background Processing

The processor now uses a background thread for disk writes, allowing:
- **Concurrent I/O and processing**: While batch N is being written to disk, batch N+1 is already being processed
- **Better resource utilization**: CPU and disk are used simultaneously
- **Faster overall processing**: No waiting for disk writes between batches

## Using Lazy Loading Datasets

### Basic Lazy Loading

```python
from shared.data_processing.lazy_dataset import LazyNPZDataset

# Create lazy dataset (data not loaded into memory yet)
train_dataset = LazyNPZDataset('data/preprocessed/train_detector.npz')

print(f"Dataset length: {len(train_dataset)}")  # Quick - just returns count

# Access individual samples (loaded on-demand)
sample = train_dataset[0]
print(sample.keys())  # dict_keys(['image', 'bboxes', 'image_path'])

# Data is only loaded when accessed
image = sample['image']
bboxes = sample['bboxes']
```

### With Caching

```python
# Cache the 100 most recently accessed samples
train_dataset = LazyNPZDataset(
    'data/preprocessed/train_detector.npz',
    cache_size=100  # LRU cache
)

# First access: loads from disk
sample1 = train_dataset[0]

# Second access: retrieved from cache (fast!)
sample1_again = train_dataset[0]
```

### PyTorch Integration

```python
import torch
from torch.utils.data import DataLoader
from shared.data_processing.lazy_dataset import LazyNPZDataset

# Create lazy dataset
dataset = LazyNPZDataset('data/preprocessed/train_detector.npz', cache_size=50)

# Use with DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Multiple workers supported
    pin_memory=True
)

# Training loop
for batch in dataloader:
    images = batch['image']
    bboxes = batch['bboxes']
    # ... your training code
```

### Combining Datasets

```python
from shared.data_processing.lazy_dataset import CombinedLazyDataset, LazyNPZDataset

# Load multiple datasets
train_ds = LazyNPZDataset('data/preprocessed/train_detector.npz')
val_ds = LazyNPZDataset('data/preprocessed/val_detector.npz')

# Combine them
combined = CombinedLazyDataset([train_ds, val_ds])

print(f"Combined length: {len(combined)}")  # Sum of both

# Access samples from either dataset
sample_from_train = combined[0]
sample_from_val = combined[len(train_ds)]  # First val sample
```

### Batch Loading

```python
dataset = LazyNPZDataset('data/preprocessed/train_detector.npz')

# Get multiple samples as a batch
indices = [0, 1, 2, 3, 4]
batch = dataset.get_batch(indices)

# batch is a dict with batched arrays
print(batch['images'].shape)  # (5, H, W, 3)
print(batch['bboxes'].shape)  # (5,) - array of bbox lists
```

## Memory Efficiency Comparison

### Old Approach (Load All)
```python
# Loads entire dataset into RAM
data = np.load('train_detector.npz')
images = data['images']  # All images in memory at once!
# Peak memory: ~10GB for 10k images
```

### New Approach (Lazy Loading)
```python
# Only metadata loaded
dataset = LazyNPZDataset('train_detector.npz')

# Only batch worth of data in memory
for i in range(0, len(dataset), 32):
    batch_indices = range(i, min(i+32, len(dataset)))
    batch = dataset.get_batch(batch_indices)
    # Peak memory: ~300MB for batch of 32
```

## Dataset Types

The lazy loader automatically detects dataset types:

### Detector Data
- Keys: `images`, `bboxes`, `image_paths`
- `bboxes[i]` is a list of bounding boxes for image `i`

### Landmarker Data
- Keys: `images`, `keypoints`, `image_paths`
- `keypoints[i]` is an array of shape (N, 3) for image `i`

### Teacher Data
- Keys: `images`, `bboxes`, `image_paths`
- `bboxes[i]` is a single bounding box (not a list)

## Best Practices

1. **Use caching for repeated access**: Set `cache_size` based on your access patterns
2. **Multiple workers**: Lazy loading works well with PyTorch's `num_workers > 0`
3. **Batch size**: Adjust based on available memory (smaller batches = less memory)
4. **Close datasets**: Call `dataset.close()` when done, or use context managers

## Performance Tips

- **Background writing during processing**: Already enabled by default
- **Cache hot samples**: Use `cache_size` for frequently accessed data
- **Memory-mapped I/O**: NPZ files are opened with `mmap_mode='r'` for efficient access
- **Parallel data loading**: Use DataLoader with `num_workers > 0`

## Example: Full Pipeline

```python
from shared.data_processing.data_processor import DataProcessor
from shared.data_processing.lazy_dataset import LazyNPZDataset
from torch.utils.data import DataLoader

# Step 1: Process raw data (with background disk writing)
processor = DataProcessor(flush_every=5)
processor.process_all()

# Step 2: Create lazy datasets
train_ds = LazyNPZDataset('data/preprocessed/train_detector.npz', cache_size=100)
val_ds = LazyNPZDataset('data/preprocessed/val_detector.npz', cache_size=50)

# Step 3: Create data loaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# Step 4: Train
for epoch in range(100):
    for batch in train_loader:
        # Only current batch is in memory!
        images = batch['image']
        bboxes = batch['bboxes']
        # ... training code
```
