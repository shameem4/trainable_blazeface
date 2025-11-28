import csv
import os

def find_csv_annotation(image_path):
    folder = os.path.dirname(image_path)
    for file in os.listdir(folder):
        if file.endswith('_annotations.csv'):
            return os.path.join(folder, file)
    return None

def decode_csv_annotation(annotation_path, image_filename):
    decoded = []
    with open(annotation_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Support both 'filename' and 'image_path' columns
            csv_filename = row.get('filename') or row.get('image_path')
            if csv_filename and os.path.basename(csv_filename) == os.path.basename(image_filename):
                try:
                    # Support both (x, y, w, h) and (xmin, ymin, xmax, ymax) formats
                    if 'xmin' in row and 'ymin' in row and 'xmax' in row and 'ymax' in row:
                        xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
                        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    elif 'x' in row and 'y' in row and 'w' in row and 'h' in row:
                        bbox = [float(row['x']), float(row['y']), float(row['w']), float(row['h'])]
                    else:
                        continue
                    decoded.append({'bbox': bbox})
                except Exception:
                    continue
    return decoded
