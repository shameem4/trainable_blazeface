"""
Evaluate MediaPipe pretrained weights on validation set.
Used for README documentation.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

from utils.model_utils import load_model


def evaluate_model(model, val_csv, data_root, max_images=1000, score_threshold=0.5, iou_threshold=0.5):
    """Evaluate model on validation set."""
    device = next(model.parameters()).device
    
    # Load validation data
    df = pd.read_csv(val_csv)
    grouped = df.groupby('image_path', sort=False)
    image_groups = list(grouped)[:max_images]
    print(f'Evaluating on {len(image_groups)} images...')

    # Metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_detections = 0
    total_gt_boxes = 0

    for image_path, group in tqdm(image_groups, desc='Evaluating'):
        # Load and preprocess image
        full_path = Path(data_root) / image_path
        img = cv2.imread(str(full_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Resize with padding to 128x128
        target_size = 128
        if orig_h >= orig_w:
            scale = target_size / orig_h
            new_h, new_w = target_size, int(orig_w * scale)
        else:
            scale = target_size / orig_w
            new_h, new_w = int(orig_h * scale), target_size
        
        resized = cv2.resize(img, (new_w, new_h))
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
        
        # Get GT boxes (x1, y1, w, h are in ABSOLUTE PIXELS)
        gt_boxes_raw = group[['x1', 'y1', 'w', 'h']].values.astype(np.float32)
        gt_boxes = []
        for x, y, w, h in gt_boxes_raw:
            # x, y, w, h are already in absolute coords in original image
            abs_y = y  # top-left y
            abs_x = x  # top-left x
            abs_h = h
            abs_w = w
            # Scale and pad to 128x128 space
            abs_y = abs_y * scale + pad_top
            abs_x = abs_x * scale + pad_left
            abs_h = abs_h * scale
            abs_w = abs_w * scale
            # Normalize to [0, 1]
            ymin = abs_y / target_size
            xmin = abs_x / target_size
            ymax = (abs_y + abs_h) / target_size
            xmax = (abs_x + abs_w) / target_size
            gt_boxes.append([ymin, xmin, ymax, xmax])
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
        
        # Process detections
        dets = detections[0].cpu().numpy() if len(detections) > 0 and len(detections[0]) > 0 else np.zeros((0, 17))
        if len(dets) > 0:
            pred_boxes = dets[:, :4]
            scores = dets[:, -1]
            mask = scores >= score_threshold
            pred_boxes = pred_boxes[mask]
        else:
            pred_boxes = np.zeros((0, 4))
        
        n_dets = len(pred_boxes)
        n_gt = len(gt_boxes)
        total_detections += n_dets
        total_gt_boxes += n_gt
        
        # Simple matching
        if n_gt == 0:
            total_fp += n_dets
        elif n_dets == 0:
            total_fn += n_gt
        else:
            matched_gt = set()
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    y1 = max(pred[0], gt[0])
                    x1 = max(pred[1], gt[1])
                    y2 = min(pred[2], gt[2])
                    x2 = min(pred[3], gt[3])
                    inter = max(0, y2 - y1) * max(0, x2 - x1)
                    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
                    area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    union = area_pred + area_gt - inter
                    iou = inter / (union + 1e-6)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
            
            total_fn += n_gt - len(matched_gt)

    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'images': len(image_groups),
        'gt_boxes': total_gt_boxes,
        'detections': total_detections,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate BlazeFace model')
    parser.add_argument('--weights', type=str, default='model_weights/blazeface.pth',
                        help='Path to model weights')
    parser.add_argument('--trained-weights', type=str, default=None,
                        help='Path to trained checkpoint (state_dict with model key)')
    parser.add_argument('--csv', type=str, default='data/splits/val.csv')
    parser.add_argument('--data-root', type=str, default='data/raw/blazeface')
    parser.add_argument('--num-images', type=int, default=500)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    args = parser.parse_args()
    
    VAL_CSV = args.csv
    DATA_ROOT = args.data_root
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load model with MediaPipe weights first
    model = load_model(args.weights, device=device)
    
    # If trained weights provided, load those on top
    if args.trained_weights:
        print(f'Loading trained weights from: {args.trained_weights}')
        checkpoint = torch.load(args.trained_weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print('  Trained weights loaded successfully')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Evaluate
    results = evaluate_model(model, VAL_CSV, DATA_ROOT, 
                             max_images=args.num_images, 
                             score_threshold=args.score_threshold)
    
    weight_type = 'Trained' if args.trained_weights else 'MediaPipe Pretrained'
    print(f'\n=== {weight_type} Weights Evaluation ===')
    print(f'Images evaluated: {results["images"]}')
    print(f'Total GT boxes: {results["gt_boxes"]}')
    print(f'Total detections: {results["detections"]}')
    print(f'True Positives: {results["tp"]}')
    print(f'False Positives: {results["fp"]}')
    print(f'False Negatives: {results["fn"]}')
    print(f'Precision: {results["precision"]:.4f}')
    print(f'Recall: {results["recall"]:.4f}')
    print(f'F1 Score: {results["f1"]:.4f}')
