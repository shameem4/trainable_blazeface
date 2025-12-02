# Trainable BlazeFace (PyTorch)

A **training-ready** implementation of Google's BlazeFace face detector in PyTorch.

This repository serves as a bridge between **[hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)** (clean PyTorch code, but inference-only) and **[FurkanOM/tf-blazeface](https://github.com/FurkanOM/tf-blazeface)** (training logic, but in TensorFlow).

It modifies the original port to enable backpropagation, adds the necessary loss functions, and implements dynamic anchor generation.

## 🌟 Key Features

* **Training Enabled:** Re-introduces `BatchNorm2d` layers and proper weight initialization (Kaiming Normal), which are stripped/folded in official TFLite inference models.
* **SSD MultiBox Loss:** Implements the complete Single Shot Detector loss function including:
    * **IoU Matching:** Dynamic matching of Ground Truth to Anchors.
    * **Hard Negative Mining:** Handles class imbalance (3:1 Negative/Positive ratio).
    * **Regression Loss:** Smooth L1 for boxes and keypoints.
* **Dynamic Anchors:** Generates anchors on-the-fly based on `config.py`, allowing support for different input resolutions (128x128 Front vs 256x256 Back).
* **Modular Design:** Separates configuration, architecture, loss, and training loop.

---

## 📂 File Structure

| File | Description |
| :--- | :--- |
| `config.py` | Configuration dictionary (kernel sizes, steps, anchor sizes) for Front/Back models. |
| `blazeface.py` | The model architecture. Modified to include **Batch Normalization** and training-friendly initialization. |
| `blazeface_anchors.py` | Generates the 896 (Front) or custom anchor boxes dynamically based on config. |
| `blazeface_loss.py` | Custom `MultiBoxLoss` class handling label encoding and Hard Negative Mining. |
| `blazeface_train.py` | The main training loop. **Contains a dummy data loader you must replace.** |

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install torch torchvision numpy
```

2. Configuration
The model behavior is defined in config.py. The default cfg_blazeface_front replicates the official Google MediaPipe "Front" (Selfie) detector settings.

```
# config.py sample
cfg_blazeface_front = {
    'min_dim': 128,
    'feature_maps': [[16, 16], [8, 8]], 
    'steps': [8, 16],
    'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    'num_keypoints': 6,
    ...
}
```

3. Training
To start training, run the main script:

```Bash

python blazeface_train.py
```

⚠️ IMPORTANT: Implementing Real Data
The file blazeface_train.py currently uses a dummy data generator (get_dummy_batch) so the script runs out-of-the-box. You must replace this with a real PyTorch DataLoader.

Target Format: Your Dataset should return:

Images: Tensor [Batch, 3, 128, 128] (Normalized)

Targets: A List of Tensors (length = Batch Size). Each tensor represents the faces in that image.

Target Tensor Structure: Shape: [Num_Faces, 16]

```Plaintext

[xmin, ymin, xmax, ymax, k1_x, k1_y, ... k6_x, k6_y]
```
Box: Standard SSD coordinates.

Keypoints: 6 facial landmarks (12 coordinates total).

Note: Ensure your coordinates match the scale expected by the logic (typically absolute pixels or normalized 0-1 depending on your specific implementation preference in blazeface_loss.py).

🧠 Technical Details
Why can't I just use the Hollance repo?
The hollance repo is a direct port of the compiled TFLite model. In TFLite, Batch Normalization layers are "folded" (mathematically merged) into the Convolutional weights to speed up inference.

Problem: You cannot train a network effectively without Batch Norm layers (gradients explode/vanish).

Solution: This repo re-inserts nn.BatchNorm2d after every convolution, making the network trainable again.

Why Dynamic Anchors?
The original inference code loads a static anchors.npy file. This is fast but inflexible. By using the AnchorGenerator (adapted from FurkanOM), you can retrain the model on different resolutions (e.g., 256x256) simply by changing min_dim in config.py, without needing to manually calculate and save new numpy arrays.

📚 Credits
Original Architecture: Google MediaPipe [https://github.com/google/mediapipe]

PyTorch Port Basis: hollance/BlazeFace-PyTorch [https://github.com/hollance/BlazeFace-PyTorch]

Training Logic & Config Structure: Adapted from FurkanOM/tf-blazeface [https://github.com/FurkanOM/tf-blazeface]