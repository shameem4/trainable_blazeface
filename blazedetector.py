import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from blazebase import BlazeBlock, FinalBlazeBlock, BlazeBase

class BlazeDetector(BlazeBase):
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    """
    
    # Type annotations for class attributes
    x_scale: float
    y_scale: float
    w_scale: float
    h_scale: float
    num_keypoints: int
    num_anchors: int
    num_coords: int
    num_classes: int
    anchors: torch.Tensor
    score_clipping_thresh: float
    min_score_thresh: float
    min_suppression_threshold: float


    def _preprocess(self, x):
        """Converts the image pixels to the range [0, 1].
        
        Note: This implementation intentionally uses [0,1] normalization instead of 
        the typical [-1,1] range. The model weights have been adapted accordingly.
        """
        return x.float() / 255.

    def predict_on_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def resize_pad(
        self,
        img: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
        """ resize and pad images to be input to the detectors

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio.

        Returns:
            img1: 256x256
            img2: 128x128
            scale: scale factor between original image and 256x256 image
            pad: pixels of padding in the original image
        """

        size0 = img.shape
        if size0[0]>=size0[1]:
            h1 = 256
            w1 = 256 * size0[1] // size0[0]
            padh = 0
            padw = 256 - w1
            scale = size0[1] / w1
        else:
            h1 = 256 * size0[0] // size0[1]
            w1 = 256
            padh = 256 - h1
            padw = 0
            scale = size0[0] / h1
        padh1 = padh//2
        padh2 = padh//2 + padh%2
        padw1 = padw//2
        padw2 = padw//2 + padw%2
        img1 = cv2.resize(img, (w1,h1))
        img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
        pad = (int(padh1 * scale), int(padw1 * scale))
        img2 = cv2.resize(img1, (128,128))
        return img1, img2, scale, pad


    def denormalize_detections(
        self,
        detections: torch.Tensor,
        scale: float,
        pad: tuple[int, int]
    ) -> torch.Tensor:
        """ maps detection coordinates from [0,1] to image coordinates

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio. This function maps the
        normalized coordinates back to the original image coordinates.

        Inputs:
            detections: nxm tensor. n is the number of detections.
                m is 4+2*k where the first 4 valuse are the bounding
                box coordinates and k is the number of additional
                keypoints output by the detector.
            scale: scalar that was used to resize the image
            pad: padding in the x and y dimensions

        """
        detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
        detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
        detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
        detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

        detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
        detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
        return detections

    def predict_on_batch(self, x: np.ndarray | torch.Tensor) -> list[torch.Tensor]:
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == self.y_scale
        assert x.shape[3] == self.x_scale

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, int(self.num_coords)+1))
            filtered_detections.append(faces)

        return filtered_detections


    def _tensors_to_detections(
        self,
        raw_box_tensor: torch.Tensor,
        raw_score_tensor: torch.Tensor,
        anchors: torch.Tensor
    ) -> list[torch.Tensor]:
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(
        self,
        raw_boxes: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(int(self.num_keypoints)):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections: torch.Tensor) -> list[torch.Tensor]:
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, self.num_coords], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = self.overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    


    # IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

    def intersect(self, box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]


    def jaccard(self, box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                  (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) *
                  (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


    def overlap_similarity(self, box: torch.Tensor, other_boxes: torch.Tensor) -> torch.Tensor:
        """Computes the IOU between a bounding box and set of other boxes."""
        return self.jaccard(box.unsqueeze(0), other_boxes).squeeze(0)