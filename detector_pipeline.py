import numpy as np
from face_detector import MTCNNFaceDetector, HaarCascadeFaceDetector
from data_process import FaceDetectionDataset

if __name__ == '__main__':
    dataset = FaceDetectionDataset('data/wider_face_split/wider_face_val_bbx_gt.txt')
    detector = MTCNNFaceDetector()
    for img,bbox in dataset.stream_data():
        detector.detect(np.asarray(img))
