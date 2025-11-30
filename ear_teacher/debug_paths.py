"""Debug image paths in validation data."""

import numpy as np
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
val_data_path = project_root / 'data/preprocessed/val_teacher.npy'

# Load data
data = np.load(val_data_path, allow_pickle=True).item()

print(f"Total validation samples: {len(data['image_paths'])}")
print(f"\nFirst 5 image paths:")
for i in range(min(5, len(data['image_paths']))):
    path = data['image_paths'][i]
    print(f"  {i}: {path}")
    # Check if file exists
    abs_path = project_root / path
    print(f"      Absolute: {abs_path}")
    print(f"      Exists: {abs_path.exists()}")

    # Try to find the file
    if not abs_path.exists():
        # Try different variations
        parts = Path(path).parts
        if 'data' in parts:
            idx = parts.index('data')
            rel_path = Path(*parts[idx:])
            test_path = project_root / rel_path
            print(f"      Alt path: {test_path}")
            print(f"      Alt exists: {test_path.exists()}")

print("\n\nLet's check what's actually in data/raw/teacher:")
teacher_dir = project_root / 'data/raw/teacher'
if teacher_dir.exists():
    for item in teacher_dir.iterdir():
        if item.is_dir():
            print(f"\n  {item.name}:")
            # Check for train/val subdirs
            for subdir in ['train', 'val', 'valid', 'test']:
                test_subdir = item / subdir
                if test_subdir.exists():
                    num_files = len(list(test_subdir.glob('*.jpg'))) + len(list(test_subdir.glob('*.png')))
                    print(f"    {subdir}/: {num_files} images")
