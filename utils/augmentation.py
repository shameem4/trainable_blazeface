"""
Image augmentation utilities for training.
"""
import cv2
import numpy as np


def augment_saturation(image: np.ndarray, factor_range: tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Apply random saturation adjustment.

    Args:
        image: RGB image
        factor_range: (min, max) saturation multiplication factor

    Returns:
        Augmented RGB image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_factor = np.random.uniform(*factor_range)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_brightness(image: np.ndarray, delta_range: tuple[float, float] = (-0.2, 0.2)) -> np.ndarray:
    """Apply random brightness adjustment.

    Args:
        image: RGB image
        delta_range: (min, max) brightness delta as fraction of 255

    Returns:
        Augmented RGB image
    """
    brightness_delta = np.random.uniform(*delta_range) * 255
    return np.clip(image.astype(np.float32) + brightness_delta, 0, 255).astype(np.uint8)


def augment_horizontal_flip(
    image: np.ndarray,
    bboxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply horizontal flip to image and bounding boxes.

    Args:
        image: RGB image
        bboxes: Bounding boxes in [ymin, xmin, ymax, xmax] format (normalized)

    Returns:
        (flipped_image, flipped_bboxes)
    """
    image = np.fliplr(image)

    # Flip x coordinates
    xmin_old = bboxes[:, 1].copy()
    xmax_old = bboxes[:, 3].copy()
    bboxes[:, 1] = 1.0 - xmax_old
    bboxes[:, 3] = 1.0 - xmin_old

    return image, bboxes


def augment_synthetic_occlusion(
    image: np.ndarray,
    num_occlusions: int = 1,
    occlusion_size_range: tuple[int, int] = (10, 50)
) -> np.ndarray:
    """Add synthetic occlusions (black rectangles) to image.

    Args:
        image: RGB image
        num_occlusions: Number of occlusions to add
        occlusion_size_range: (min, max) size of occlusion rectangles

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]

    for _ in range(num_occlusions):
        occ_h = np.random.randint(*occlusion_size_range)
        occ_w = np.random.randint(*occlusion_size_range)
        occ_y = np.random.randint(0, max(1, h - occ_h))
        occ_x = np.random.randint(0, max(1, w - occ_w))

        image[occ_y:occ_y + occ_h, occ_x:occ_x + occ_w] = 0

    return image
