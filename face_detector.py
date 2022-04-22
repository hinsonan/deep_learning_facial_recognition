import cv2
from facenet_pytorch import MTCNN
import numpy as np
import torch

class HaarCascadeFaceDetector:

    def __init__(self) -> None:
        self.model: cv2.CascadeClassifier = cv2.CascadeClassifier('models/detector/haarcascade_frontalface_default.xml')

    def detect(self,img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = self.model.detectMultiScale(gray_img,1.03,1,minSize=[50,50])

        # Draw rectangle around the faces
        print(len(faces))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        cv2.imshow('img', img)
        cv2.waitKey()

class MTCNNFaceDetector:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True,device=self.device)

    def detect(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes, _ = self.model.detect(img)
        for box in boxes:
            # clip the bounding box dimensions to the image size
            box = np.clip(box,0,img.shape[0]-2)
            box = [int(x) for x in box]
            print(box)
            cv2.rectangle(img,(box[0],0),(box[2],box[3]),(255, 0, 0), 2)
        cv2.imshow('img', img)
        cv2.waitKey()

if __name__ == '__main__':
    # detector = HaarCascadeFaceDetector()
    detector = MTCNNFaceDetector()
    img = cv2.imread('data/greyscale_faces/anger/372662640_e8dc799d8b_b_face.png')
    detector.detect(img)
