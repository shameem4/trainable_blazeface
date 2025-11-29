"""
ONNX inference script for image segmentation model.

This script loads an ONNX model and performs inference on an input image to generate
an alpha mask. The mask is combined with the RGB image and displayed side by side.
"""

import onnxruntime as ort
import cv2
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from torchvision.transforms.functional import normalize
from tkinter import filedialog
import tkinter as tk


INPUT_SIZE = [1200, 1800]

def keep_large_components(a: np.ndarray) -> np.ndarray:
    """Remove small connected components from a binary mask, keeping only large regions.

    Args:
        a: Input binary mask as numpy array of shape (H,W) or (H,W,1)

    Returns:
        Processed mask with only large connected components remaining, shape (H,W,1)
    """
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))
    a_mask = (a > 25).astype(np.uint8) * 255

    # Apply the Component analysis function
    analysis = cv2.connectedComponentsWithStats(a_mask, connectivity=4, ltype=cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Find the components to be kept
    h, w = a.shape[:2]
    area_limit = 50000 * (h * w) / (INPUT_SIZE[1] * INPUT_SIZE[0])
    i_to_keep = []
    for i in range(1, totalLabels):
        area = values[i, cv2.CC_STAT_AREA]
        if area > area_limit:
            i_to_keep.append(i)

    if len(i_to_keep) > 0:
        # Or masks to be kept
        final_mask = np.zeros_like(a, dtype=np.uint8)
        for i in i_to_keep:
            componentMask = (label_ids == i).astype("uint8") * 255
            final_mask = cv2.bitwise_or(final_mask, componentMask)

        # Remove other components
        # Keep edges
        final_mask = cv2.dilate(final_mask, dilate_kernel, iterations = 2)
        a = cv2.bitwise_and(a, final_mask)
        a = a.reshape((a.shape[0], a.shape[1], 1))
        
    return a

def read_img(img: Path) -> np.ndarray:
    """Read an image from a local path.

    Args:
        img: File path to image

    Returns:
        Image as numpy array in RGB format with shape (H,W,3)
    """
    im = cv2.imread(str(img))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def preprocess_input(im: np.ndarray) -> torch.Tensor:
    """Preprocess image for model input.

    Args:
        im: Input image as numpy array of shape (H,W,C)

    Returns:
        Preprocessed image as normalized torch tensor of shape (1,3,H,W)
    """
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]

    if im.shape[2] == 4:  # if image has alpha channel, remove it
        im = im[:,:,:3]

    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), INPUT_SIZE, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor,255.0)
    image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

    if torch.cuda.is_available():
        image=image.cuda()

    return image

def postprocess_output(result: np.ndarray, orig_im_shape: Tuple[int, int]) -> np.ndarray:
    """Postprocess ONNX model output.

    Args:
        result: Model output as numpy array of shape (1,1,H,W)
        orig_im_shape: Original image dimensions (height, width)

    Returns:
        Processed binary mask as numpy array of shape (H,W,1)
    """
    result_tensor = torch.squeeze(F.interpolate(
        torch.from_numpy(result).unsqueeze(0), (orig_im_shape), mode='bilinear'), 0)
    ma = torch.max(result_tensor)
    mi = torch.min(result_tensor)
    result_tensor = (result_tensor-mi)/(ma-mi)

    # a is alpha channel. 255 means foreground, 0 means background.
    a = (result_tensor*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)

    # postprocessing
    a = keep_large_components(a)

    return a

def process_and_display_image(src: Path, ort_session: Any) -> None:
    """Process an image through ONNX model and display original vs processed side by side.

    Args:
        src: Source image path
        ort_session: ONNX runtime inference session

    Returns:
        None
    """
    # Load and preprocess image
    image_orig = read_img(src)
    image = preprocess_input(image_orig)

    # Prepare ONNX input (move to CPU if on CUDA)
    inputs: Dict[str, Any] = {ort_session.get_inputs()[0].name: image.cpu().numpy()}

    # Get ONNX output and post-process
    result = ort_session.run(None, inputs)[0][0]
    alpha = postprocess_output(result, (image_orig.shape[0], image_orig.shape[1]))

    # Create processed image with alpha mask
    img_w_alpha = np.dstack((image_orig, alpha))

    # Convert for display (original is RGB, processed needs RGBA)
    original_bgr = cv2.cvtColor(image_orig, cv2.COLOR_RGB2BGR)
    processed_bgra = cv2.cvtColor(img_w_alpha, cv2.COLOR_RGBA2BGRA)

    # Create white background for processed image
    white_bg = np.ones_like(original_bgr) * 255
    alpha_norm = alpha.astype(float) / 255.0

    # Blend processed image with white background
    processed_with_bg = processed_bgra[:, :, :3] * alpha_norm + white_bg * (1 - alpha_norm)
    processed_with_bg = processed_with_bg.astype(np.uint8)

    # Resize images to fit screen if needed
    max_height = 800
    if original_bgr.shape[0] > max_height:
        scale = max_height / original_bgr.shape[0]
        new_width = int(original_bgr.shape[1] * scale)
        new_height = int(original_bgr.shape[0] * scale)
        original_bgr = cv2.resize(original_bgr, (new_width, new_height))
        processed_with_bg = cv2.resize(processed_with_bg, (new_width, new_height))

    # Create side-by-side comparison
    comparison = np.hstack([original_bgr, processed_with_bg])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, 'Background Removed', (original_bgr.shape[1] + 10, 30),
                font, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow('Background Removal Comparison', comparison)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def choose_file() -> Path:
    """Open file dialog to choose an image file.

    Returns:
        Path to selected image file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        raise ValueError("No file selected")

    return Path(file_path)

if __name__ == "__main__":
    MODEL_PATH = "models/background_removal_model.onnx"

    # Initialize ONNX runtime session with CUDA and CPU providers
    print("Initializing ONNX model...")
    ort_session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Open file dialog to choose image
    try:
        image_path = choose_file()
        print(f"Processing: {image_path}")

        # Process and display the image
        process_and_display_image(image_path, ort_session)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error processing image: {e}")
