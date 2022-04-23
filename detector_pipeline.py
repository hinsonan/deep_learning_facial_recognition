import json
import numpy as np
from face_detector import MTCNNFaceDetector, HaarCascadeFaceDetector
from data_process import FaceDetectionDataset
from PIL import ImageDraw

class DetectorPipeline:
    
    def __init__(self,backbone_model='MTCNN') -> None:
        self.dataset = FaceDetectionDataset('data/wider_face_split/wider_face_val_bbx_gt.txt')
        if backbone_model == 'MTCNN':
            self.detector = MTCNNFaceDetector()
        else:
            self.detector = HaarCascadeFaceDetector()

    def display_detection(self,img,bbox,truth_bbox=None):
        pass

    def _get_iou_metrics(self,bb1,bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        assert bb1[0] <= bb1[2]
        assert bb1[1] <= bb1[3]
        assert bb2[0] <= bb2[2]
        assert bb2[1] <= bb2[3]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
    
    def gather_metrics(self):
        metrics = {'Accuracy IoU at 50%':None,'Accuracy IoU at 75%':None, 'Accuracy IoU at 85%':None}
        IoU_50 = []
        IoU_75 = []
        IoU_85 = []
        for img,bbox in self.dataset.stream_data():
            truth_box = [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]
            boxes = self.detector.detect(np.asarray(img))
            # if no boxes returns then the model failed to pick up the face in the picture
            if type(boxes) != np.ndarray and type(boxes) != list:
                IoU_50.append(0)
                IoU_75.append(0)
                IoU_85.append(0)
                continue
            for box in boxes:
                # clip the bounding box dimensions to the image size
                box = np.clip(box,0,img.size[0]-2)
                box = [int(x) for x in box]
                IoU = self._get_iou_metrics(box,truth_box)
                IoU_50.append(1 if IoU >= .5 else 0)
                IoU_75.append(1 if IoU >= .75 else 0)
                IoU_85.append(1 if IoU >= .85 else 0)
        metrics['Accuracy IoU at 50%'] = IoU_50.count(1) / len(IoU_50)
        metrics['Accuracy IoU at 75%'] = IoU_75.count(1) / len(IoU_75)
        metrics['Accuracy IoU at 85%'] = IoU_85.count(1) / len(IoU_85)

        with open(f'experiment_results/detection_metrics/{type(self.detector)}.json','w') as f:
            json.dump(metrics,f,indent=4)

    def run_pipeline(self):
        self.gather_metrics()

if __name__ == '__main__':
    pipeline = DetectorPipeline(backbone_model='MTCNN')
    pipeline.run_pipeline()
    pipeline = DetectorPipeline(backbone_model='Cascade')
    pipeline.run_pipeline()