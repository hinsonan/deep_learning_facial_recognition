import cv2
from facenet_pytorch import MTCNN
import torch
import os

class HaarCascadeFaceDetector:

    def __init__(self) -> None:
        self.model: cv2.CascadeClassifier = cv2.CascadeClassifier(f'models{os.sep}detector{os.sep}haarcascade_frontalface_default.xml')

    def detect(self,img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = self.model.detectMultiScale(gray_img,1.03,1,minSize=[50,50])

        boxes = []
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            box = [int(x), int(y), int(x+w), int(y+h)]
            boxes.append(box)
        return boxes
class MTCNNFaceDetector:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = MTCNN(keep_all=True,device=self.device)

    def detect(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes, _ = self.model.detect(img)
        return boxes

if __name__ == '__main__':
    # detector = HaarCascadeFaceDetector()
    detector = MTCNNFaceDetector()
    img = cv2.imread(f'data{os.sep}greyscale_faces{os.sep}anger{os.sep}372662640_e8dc799d8b_b_face.png')
    detector.detect(img)
