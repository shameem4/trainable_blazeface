import json
import os

def find_coco_annotation(image_path):
    folder = os.path.dirname(image_path)
    for file in os.listdir(folder):
        if file.endswith('_annotations.coco.json'):
            return os.path.join(folder, file)
    return None

def decode_coco_annotation(annotation_path, image_filename):
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    img_id = None
    for img in annotation.get('images', []):
        if os.path.basename(img['file_name']) == os.path.basename(image_filename):
            img_id = img['id']
            break
    if img_id is None:
        return []
    decoded = []
    for ann in annotation.get('annotations', []):
        if ann['image_id'] == img_id:
            item = {'bbox': ann.get('bbox'), 'keypoints': ann.get('keypoints')}
            decoded.append(item)
    return decoded
