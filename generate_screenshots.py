"""
Generate screenshots for README documentation.
Creates before/after comparison images showing detection results.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

from utils.model_utils import load_model


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Draw detection boxes on image."""
    img = image.copy()
    h, w = img.shape[:2]
    
    for det in detections:
        if len(det) >= 4:
            # Detections are [ymin, xmin, ymax, xmax] normalized
            ymin, xmin, ymax, xmax = det[:4]
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    return img


def draw_gt_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """Draw ground truth boxes on image. Boxes are [x, y, w, h] absolute."""
    img = image.copy()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
    return img


def run_inference(model, image, device):
    """Run inference on an image."""
    # Resize with padding to 128x128
    h, w = image.shape[:2]
    target_size = 128
    
    if h >= w:
        scale = target_size / h
        new_h, new_w = target_size, int(w * scale)
    else:
        scale = target_size / w
        new_h, new_w = int(h * scale), target_size
    
    resized = cv2.resize(image, (new_w, new_h))
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    
    padded = cv2.copyMakeBorder(resized, pad_top, target_size - new_h - pad_top,
                                 pad_left, target_size - new_w - pad_left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # To tensor
    img_tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Run inference
    with torch.no_grad():
        detections = model.predict_on_batch(img_tensor)
    
    # Scale detections back to original image coordinates
    scaled_dets = []
    for det in detections:
        if len(det) >= 4:
            ymin, xmin, ymax, xmax = det[:4]
            # Remove padding offset and scale
            ymin_px = (ymin * target_size - pad_top) / scale
            xmin_px = (xmin * target_size - pad_left) / scale
            ymax_px = (ymax * target_size - pad_top) / scale
            xmax_px = (xmax * target_size - pad_left) / scale
            # Back to normalized
            scaled_dets.append([ymin_px / h, xmin_px / w, ymax_px / h, xmax_px / w] + list(det[4:]))
    
    return scaled_dets


def create_comparison_image(pretrained_model, trained_model, image_path, gt_boxes, device, output_path):
    """Create side-by-side comparison of pretrained vs trained detections."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference with both models
    pretrained_dets = run_inference(pretrained_model, img_rgb, device)
    trained_dets = run_inference(trained_model, img_rgb, device)
    
    # Draw on images
    img_pretrained = draw_gt_boxes(img.copy(), gt_boxes, color=(128, 128, 128), thickness=1)
    img_pretrained = draw_detections(img_pretrained, pretrained_dets, color=(0, 255, 0), thickness=2)
    
    img_trained = draw_gt_boxes(img.copy(), gt_boxes, color=(128, 128, 128), thickness=1)
    img_trained = draw_detections(img_trained, trained_dets, color=(0, 255, 0), thickness=2)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_pretrained, f'MediaPipe Pretrained ({len(pretrained_dets)} detections)', 
                (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img_trained, f'Fine-tuned ({len(trained_dets)} detections)', 
                (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(img_pretrained, f'GT: {len(gt_boxes)} faces', 
                (10, 60), font, 0.5, (128, 128, 128), 1)
    cv2.putText(img_trained, f'GT: {len(gt_boxes)} faces', 
                (10, 60), font, 0.5, (128, 128, 128), 1)
    
    # Combine side by side
    combined = np.hstack([img_pretrained, img_trained])
    
    # Save
    cv2.imwrite(str(output_path), combined)
    print(f'Saved: {output_path}')
    return True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Create output directory
    output_dir = Path('assets/screenshots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print('Loading pretrained model...')
    pretrained_model = load_model('model_weights/blazeface.pth', device=device)
    
    print('Loading trained model...')
    trained_model = load_model('model_weights/blazeface.pth', device=device)
    checkpoint = torch.load('runs/checkpoints/BlazeFace_best.pth', map_location=device)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.eval()
    
    # Load validation data
    val_csv = 'data/splits/val.csv'
    data_root = Path('data/raw/blazeface')
    df = pd.read_csv(val_csv)
    
    # Find images with multiple faces (more interesting for comparison)
    grouped = df.groupby('image_path', sort=False)
    multi_face_images = [(path, group) for path, group in grouped if len(group) >= 3]
    
    print(f'Found {len(multi_face_images)} images with 3+ faces')
    
    # Generate screenshots for a few good examples
    count = 0
    for image_path, group in multi_face_images[:20]:  # Check first 20
        full_path = data_root / image_path
        if not full_path.exists():
            continue
        
        gt_boxes = group[['x1', 'y1', 'w', 'h']].values.tolist()
        
        output_path = output_dir / f'comparison_{count + 1}.jpg'
        success = create_comparison_image(
            pretrained_model, trained_model, 
            full_path, gt_boxes, device, output_path
        )
        
        if success:
            count += 1
            if count >= 5:  # Generate 5 comparison images
                break
    
    print(f'\nGenerated {count} comparison screenshots in {output_dir}')
    print('\nTo add to README, use:')
    print('![Before vs After](assets/screenshots/comparison_1.jpg)')


if __name__ == '__main__':
    main()
