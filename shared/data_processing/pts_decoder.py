import os

def find_pts_annotation(image_path):
    """Find corresponding .pts file for an image."""
    base_path = os.path.splitext(image_path)[0]
    pts_path = base_path + '.pts'
    if os.path.exists(pts_path):
        return pts_path
    return None

def decode_pts_annotation(annotation_path, image_filename):
    """
    Decode .pts annotation file.

    PTS file format:
    version: 1
    n_points: N
    {
    x1 y1
    x2 y2
    ...
    }

    Returns list with single dict containing keypoints in COCO format [x, y, visibility, ...]
    """
    decoded = []
    keypoints = []

    try:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        in_points = False
        for line in lines:
            line = line.strip()

            if line == '{':
                in_points = True
                continue
            elif line == '}':
                in_points = False
                break

            if in_points:
                parts = line.split()
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    # COCO format: [x, y, visibility, ...]
                    # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
                    keypoints.extend([x, y, 2])

        if keypoints:
            decoded.append({'keypoints': keypoints, 'bbox': None})

    except Exception as e:
        print(f"Error decoding PTS file: {e}")
        return []

    return decoded
