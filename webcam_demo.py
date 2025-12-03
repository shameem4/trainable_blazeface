import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from typing import Sequence


from blazeface import BlazeFace
from blazeface_landmark import BlazeFaceLandmark


def draw_detections(
    img: np.ndarray,
    detections: torch.Tensor | np.ndarray,
    with_keypoints: bool = True
) -> None:
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        x1=xmin
        x2=xmin
        x3=xmax
        x4=xmax
        y1=ymin
        y2=ymax
        y3=ymin
        y4=ymax

        box=(x1,x2,x3,x4), (y1,y2,y3,y4)
        boxes=[box]
        draw_boxes(img, boxes, color=(255, 0, 0),thickness=2)

        points=[]
        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                points.append((kp_x,kp_y))
            draw_circles(img, points, color=(0, 0, 255), size=2)


def draw_boxes(
    img: np.ndarray,
    boxes: Sequence[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]],
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2
) -> None:
    """Draw rotated bounding boxes from ROI corner points.
    
    Note: This function expects boxes as corner point tuples ((x1,x2,x3,x4), (y1,y2,y3,y4))
    from the detection2roi output, not standard axis-aligned bboxes. The line connections
    are intentional for drawing rotated/oriented bounding boxes.
    """
    for box in boxes:
        (x1,x2,x3,x4), (y1,y2,y3,y4) = box
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), color, thickness)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), color, thickness)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), color, thickness)


def draw_circles(
    img: np.ndarray,
    points: Sequence[tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    size: int = 2
) -> None:
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=size)


if __name__ == "__main__":
    # Get the directory where this script is located for relative model paths
    SCRIPT_DIR = Path(__file__).parent

    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    face_detector = BlazeFace(back_model=False).to(gpu)
    face_detector.load_weights(str(SCRIPT_DIR / "model_weights" / "blazeface.pth"))

    face_regressor = BlazeFaceLandmark().to(gpu)
    face_regressor.load_weights(str(SCRIPT_DIR / "model_weights" / "blazeface_landmark.pth"))

    WINDOW = "test"
    cv2.namedWindow(WINDOW)
    capture = cv2.VideoCapture(0)
    mirror_img = True
    if capture.isOpened():
        hasFrame, frame = capture.read()
        frame_ct = 0
        while hasFrame:
            frame_ct +=1
            if mirror_img:
                frame = np.ascontiguousarray(frame[:,::-1,::-1])
            else:
                frame = np.ascontiguousarray(frame[:,:,::-1])


            face_detections = face_detector.process(frame)

            draw_detections(frame, face_detections)


            landmarks,boxes = face_regressor.process(frame, face_detections)

            if len(landmarks):
                draw_circles(frame, landmarks[0][:,:2],  color=(0, 255, 0), size=2)
           
                      
            draw_boxes(frame, boxes, color=(0, 0, 0),thickness=2)



            cv2.imshow(WINDOW, frame[:,:,::-1])

            hasFrame, frame = capture.read()
            key = cv2.waitKey(1)
            if key == 27 or key ==ord('q'):
                break            
    capture.release()
    cv2.destroyAllWindows()
    sys.exit(0)