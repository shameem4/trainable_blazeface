"""
Utility for drawing annotations (bounding boxes and keypoints) on images.
"""
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def draw_bounding_boxes(draw, annotations, color='red', width=2):
    """
    Draw bounding boxes on an image.

    Args:
        draw: PIL ImageDraw object
        annotations: List of annotation dictionaries with 'bbox' key
        color: Color for the bounding box outline (default: 'red')
        width: Line width for the bounding box (default: 2)
    """
    for ann in annotations:
        if 'bbox' in ann and ann['bbox']:
            x, y, w, h = ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def draw_keypoints(draw, annotations, color='blue', radius=2):
    """
    Draw keypoints on an image.

    Args:
        draw: PIL ImageDraw object
        annotations: List of annotation dictionaries with 'keypoints' key
                    Keypoints format: [x, y, visibility, x, y, visibility, ...]
        color: Color for the keypoints (default: 'blue')
        radius: Radius of the keypoint circles (default: 2)
    """
    for ann in annotations:
        if 'keypoints' in ann and ann['keypoints']:
            kps = ann['keypoints']
            for i in range(0, len(kps), 3):
                xk, yk, v = kps[i:i+3]
                if v > 0:  # Only draw visible keypoints
                    draw.ellipse(
                        [xk - radius, yk - radius, xk + radius, yk + radius],
                        fill=color
                    )


def draw_annotations_on_image(image_path, annotations, bbox_color='red', keypoint_color='blue',
                              bbox_width=2, keypoint_radius=2):
    """
    Draw annotations (bounding boxes and keypoints) on an image.

    Args:
        image_path: Path to the image file
        annotations: List of annotation dictionaries with 'bbox' and/or 'keypoints' keys
        bbox_color: Color for bounding boxes (default: 'red')
        keypoint_color: Color for keypoints (default: 'blue')
        bbox_width: Line width for bounding boxes (default: 2)
        keypoint_radius: Radius of keypoint circles (default: 2)

    Returns:
        PIL Image object with annotations drawn
    """
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # Draw bounding boxes
    draw_bounding_boxes(draw, annotations, color=bbox_color, width=bbox_width)

    # Draw keypoints
    draw_keypoints(draw, annotations, color=keypoint_color, radius=keypoint_radius)

    return image


def display_image(image, figsize=(8, 8)):
    """
    Display an image using matplotlib.

    Args:
        image: PIL Image object
        figsize: Figure size tuple (width, height) in inches (default: (8, 8))
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def visualize_annotations(image_path, annotations, bbox_color='red', keypoint_color='blue',
                         bbox_width=2, keypoint_radius=2, figsize=(8, 8)):
    """
    Draw and display annotations on an image.

    Args:
        image_path: Path to the image file
        annotations: List of annotation dictionaries with 'bbox' and/or 'keypoints' keys
        bbox_color: Color for bounding boxes (default: 'red')
        keypoint_color: Color for keypoints (default: 'blue')
        bbox_width: Line width for bounding boxes (default: 2)
        keypoint_radius: Radius of keypoint circles (default: 2)
        figsize: Figure size tuple (width, height) in inches (default: (8, 8))
    """
    image = draw_annotations_on_image(
        image_path, annotations,
        bbox_color=bbox_color,
        keypoint_color=keypoint_color,
        bbox_width=bbox_width,
        keypoint_radius=keypoint_radius
    )
    display_image(image, figsize=figsize)
