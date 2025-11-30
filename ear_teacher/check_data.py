"""Check the structure of validation data."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

val_data_path = '../data/preprocessed/val_teacher.npy'

# Load data
data = np.load(val_data_path, allow_pickle=True)

print(f"Data type: {type(data)}")
print(f"Data dtype: {data.dtype}")
print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

if data.dtype == object:
    data_content = data.item()
    print(f"Content type: {type(data_content)}")
    if isinstance(data_content, dict):
        print(f"Dictionary keys: {data_content.keys()}")
        for key in data_content.keys():
            val = data_content[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            elif isinstance(val, list):
                print(f"  {key}: list with {len(val)} items")
                if len(val) > 0:
                    print(f"    First item: {val[0]}")
else:
    print(f"Direct array shape: {data.shape}")
    print(f"Direct array dtype: {data.dtype}")
