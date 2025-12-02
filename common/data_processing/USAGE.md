# Data Processing Usage Guide

## Overview

The data processor now uses a **metadata-only approach** with NPY files:
- **NPY files contain only metadata**: Image paths + annotations (bboxes/keypoints)
- **Images stay on disk**: Loaded on-demand during training (lazy loading)
- **Background disk writing**: Processing continues while metadata is flushed to disk
- **10-100x faster preprocessing**: No image loading/saving during preprocessing
- **10-100x smaller files**: ~1-10MB instead of ~1-10GB
- **No storage duplication**: One copy of images on disk

## Processing Data

### Basic Usage

```python
from common.data_processing.data_processor import DataProcessor

# Create processor
processor = DataProcessor(
    raw_data_dir='common/data/raw',
    output_dir='common/data/preprocessed',
    batch_size=500,         # Annotations per batch (very fast now!)
    num_workers=10,         # Parallel workers
    flush_every=1           # Flush every N batches
)

# Process all datasets (creates .npy metadata files)
processor.process_all(train_split=0.8, include_teacher=True)
```

### Command Line

```bash
# Process all datasets
python -m shared.data_processing.data_processor --all

# Process specific datasets
python -m shared.data_processing.data_processor --detector --landmarker

# Custom settings (much faster with metadata-only!)
python -m shared.data_processing.data_processor --all --batch-size 1000 --workers 10
```

### What Gets Created

After processing, you'll have small NPY metadata files:

```
data/preprocessed/
├── train_detector.npy      # ~1-5MB (was ~5GB with images!)
├── val_detector.npy        # ~500KB
├── train_landmarker.npy    # ~2MB
├── val_landmarker.npy      # ~800KB
├── train_teacher.npy       # ~1MB
└── val_teacher.npy         # ~400KB
```

Each NPY file contains:
- `image_paths`: Array of paths to images on disk
- `bboxes` or `keypoints`: Annotation data

### Background Processing

The processor uses a background thread for disk writes:
- **Concurrent I/O and processing**: While batch N is being written to disk, batch N+1 is already being processed
- **No image loading**: Only parses annotations (very fast!)
- **Better resource utilization**: CPU and disk used simultaneously

## Using Lazy Loading Datasets

### Basic Lazy Loading

```python
from common.data_processing.lazy_dataset import LazyNPYDataset

# Create lazy dataset (loads metadata only - instant!)
train_dataset = LazyNPYDataset('common/data/preprocessed/train_detector.npy')

print(f"Dataset length: {len(train_dataset)}")  # Instant - just returns count

# Access individual samples (images loaded on-demand from disk)
sample = train_dataset[0]
print(sample.keys())  # dict_keys(['image', 'bboxes', 'image_path'])

# Image is loaded from disk only when accessed
image = sample['image']        # Loaded from disk now!
bboxes = sample['bboxes']      # From metadata
path = sample['image_path']    # From metadata
```

### With Caching

```python
# Cache the 100 most recently accessed samples
train_dataset = LazyNPYDataset(
    'common/data/preprocessed/train_detector.npy',
    cache_size=100  # LRU cache for loaded images
)

# First access: loads image from disk
sample1 = train_dataset[0]  # Disk I/O

# Second access: retrieved from cache (fast!)
sample1_again = train_dataset[0]  # From cache - instant!
```

### PyTorch Integration

```python
import torch
from torch.utils.data import DataLoader
from common.data_processing.lazy_dataset import LazyNPYDataset

# Create lazy dataset
dataset = LazyNPYDataset(
    'common/data/preprocessed/train_detector.npy',
    cache_size=50,          # Cache 50 loaded images
    image_loader='pil'      # or 'cv2'
)

# Use with DataLoader (images loaded in parallel!)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # Parallel image loading
    pin_memory=True
)

# Training loop - only current batch in memory!
for batch in dataloader:
    images = batch['image']        # (32, H, W, 3) - just loaded from disk
    bboxes = batch['bboxes']       # (32,) array of bbox lists
    paths = batch['image_path']    # (32,) array of paths
    # ... your training code
```

### Teacher Data (Auto-Cropping)

Teacher datasets automatically crop images using bboxes:

```python
# Teacher dataset
teacher_ds = LazyNPYDataset('common/data/preprocessed/train_teacher.npy')

sample = teacher_ds[0]
# sample['image'] is already cropped to the ear!
# sample['bbox'] contains the crop coordinates
```

### Combining Datasets

```python
from common.data_processing.lazy_dataset import CombinedLazyDataset, LazyNPYDataset

# Load multiple datasets
train_ds = LazyNPYDataset('common/data/preprocessed/train_detector.npy')
val_ds = LazyNPYDataset('common/data/preprocessed/val_detector.npy')

# Combine them
combined = CombinedLazyDataset([train_ds, val_ds])

print(f"Combined length: {len(combined)}")  # Sum of both

# Access samples from either dataset
sample_from_train = combined[0]
sample_from_val = combined[len(train_ds)]  # First val sample
```

### Metadata Only (No Image Loading)

Get annotations without loading images:

```python
dataset = LazyNPYDataset('common/data/preprocessed/train_detector.npy')

# Get metadata only (no image I/O)
metadata = dataset.get_metadata(0)
# metadata = {'image_path': '...', 'bboxes': [...]}
```

## Memory Efficiency Comparison

### Old Approach (Images in NPZ)
```python
# Loads entire dataset into RAM
data = np.load('train_detector.npz')
images = data['images']  # All 10k images in memory at once!
# Peak memory: ~10GB for 10k images
# File size: ~10GB on disk
# Preprocessing time: 30+ minutes (loading/saving images)
```

### New Approach (Metadata NPY + Lazy Loading)
```python
# Only metadata loaded (instant)
dataset = LazyNPYDataset('train_detector.npy')

# Only current batch in memory
for i in range(0, len(dataset), 32):
    batch_indices = range(i, min(i+32, len(dataset)))
    batch = dataset.get_batch(batch_indices)
    # Peak memory: ~300MB for batch of 32
    # File size: ~2MB on disk
    # Preprocessing time: 30 seconds (just parsing annotations!)
```

## Dataset Types

The lazy loader automatically detects dataset types:

### Detector Data
- Keys: `image_paths`, `bboxes`
- `bboxes[i]` is a list of bounding boxes for image `i`
- Images loaded full-size

### Landmarker Data
- Keys: `image_paths`, `keypoints`
- `keypoints[i]` is an array of shape (N, 3) for image `i`
- Images loaded full-size

### Teacher Data
- Keys: `image_paths`, `bboxes`
- `bboxes[i]` is a single bounding box (not a list)
- **Images automatically cropped** to bbox when loaded

## Best Practices

1. **Use caching for repeated access**: Set `cache_size` based on your access patterns
2. **Multiple workers**: Use DataLoader with `num_workers > 0` for parallel image loading
3. **Adjust batch size**: Based on image size and memory (no more huge datasets in RAM!)
4. **Fast preprocessing**: Metadata-only processing is 10-100x faster
5. **No duplication**: Keep one copy of images, reference them from NPY files

## Performance Tips

- **Background writing during processing**: Already enabled by default
- **Cache hot samples**: Use `cache_size` for frequently accessed data
- **Parallel loading**: Use DataLoader with `num_workers > 0`
- **Fast iteration**: Modify annotations without reprocessing images
- **Small git repos**: NPY files are tiny, easy to version control

## Example: Full Pipeline

```python
from common.data_processing.data_processor import DataProcessor
from common.data_processing.lazy_dataset import LazyNPYDataset
from torch.utils.data import DataLoader

# Step 1: Process raw data (metadata only - very fast!)
processor = DataProcessor(flush_every=1, batch_size=1000)
processor.process_all()
# Completes in ~30 seconds instead of 30 minutes!

# Step 2: Create lazy datasets
train_ds = LazyNPYDataset(
    'common/data/preprocessed/train_detector.npy',
    cache_size=100,
    image_loader='pil'
)
val_ds = LazyNPYDataset('common/data/preprocessed/val_detector.npy', cache_size=50)

# Step 3: Create data loaders
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel image loading
    pin_memory=True
)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# Step 4: Train
for epoch in range(100):
    for batch in train_loader:
        # Only current batch is in memory!
        # Images loaded in parallel by workers
        images = batch['image']
        bboxes = batch['bboxes']
        # ... training code
```

## Advantages Over Old Approach

| Aspect | Old (Images in NPZ) | New (Metadata NPY) |
|--------|---------------------|-------------------|
| File size | ~10GB | ~2MB |
| Preprocessing time | 30+ minutes | 30 seconds |
| Memory usage | All in RAM | Only batch in RAM |
| Storage duplication | Yes (original + NPZ) | No (one copy) |
| Flexibility | Hard to update | Easy to update |
| Git-friendly | No (huge files) | Yes (tiny files) |
| Training speed | Same | Same |

## Migration from NPZ

If you have old NPZ files, the new lazy loader is incompatible. Simply reprocess your data:

```bash
# Reprocess with new metadata-only approach
python -m shared.data_processing.data_processor --all
```

This will create new NPY metadata files in minutes!
