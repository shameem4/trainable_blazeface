import os
import tkinter as tk
from tkinter import filedialog

# Import unified decoder and image drawing utility (support both standalone and module usage)
try:
    from shared.data_decoder.decoder import find_annotation, decode_annotation, get_annotation_color
    from shared.image_processing.annotation_drawer import visualize_annotations
except ImportError:
    import sys
    import pathlib
    script_dir = pathlib.Path(__file__).parent.resolve()
    shared_dir = script_dir.parent / 'shared'
    sys.path.insert(0, str(shared_dir / 'data_decoder'))
    sys.path.insert(0, str(shared_dir / 'image_processing'))
    from decoder import find_annotation, decode_annotation, get_annotation_color  # type: ignore
    from annotation_drawer import visualize_annotations  # type: ignore





# Draw annotations on image (COCO, CSV, or PTS)
def draw_annotations(image_path, annotation_path, annotation_type):
    # Use unified decoder
    decoded = decode_annotation(annotation_path, image_path, annotation_type)
    bbox_color = get_annotation_color(annotation_type)

    if not decoded:
        print('Image not found in annotation file.')
        return

    # Use annotation drawer utility to visualize
    visualize_annotations(
        image_path,
        decoded,
        bbox_color=bbox_color,
        keypoint_color='blue',
        bbox_width=2,
        keypoint_radius=2,
        figsize=(8, 8)
    )


def main():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title='Select image file', filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
    if not image_path:
        print('No image selected.')
        return
    annotation_path, annotation_type = find_annotation(image_path)
    print(annotation_path,"\n", annotation_type,"\n", image_path,"\n")
    if not annotation_path:
        print('No annotation file found (COCO, CSV, or PTS).')
        return
    draw_annotations(image_path, annotation_path, annotation_type)

if __name__ == '__main__':
    main()
