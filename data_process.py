import os
import re
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch import Tensor

class FaceDataset(Dataset):

    GREYSCALE_DATA = f'data{os.sep}greyscale_faces'
    #RGB_DATA = f'data{os.sep}rgb_faces'

    def __init__(self,transform=None) -> None:
        self.label_encoding = self._get_label_encodings()
        self.data_count = self._get_count()
        self.transform = transform
        self.image_files, self.labels = self._get_all_image_paths_and_labels()

    def _get_label_encodings(self):
        dir_list = os.listdir(os.path.join(FaceDataset.GREYSCALE_DATA))
        encoder = LabelEncoder()
        encoder.fit(dir_list)
        return encoder

    def _get_all_image_paths_and_labels(self):
        image_files = []
        labels = []
        dirs = os.listdir(FaceDataset.GREYSCALE_DATA)
        for dir in dirs:
            file_list = os.listdir(os.path.join(FaceDataset.GREYSCALE_DATA,dir))
            for idx,f_string in enumerate(file_list):
                file_list[idx] = os.path.join(FaceDataset.GREYSCALE_DATA,dir,f_string)
            image_files.append(file_list)
            labels.append(self.label_encoding.transform(np.array([dir]).repeat(len(file_list))))
        # faltten
        image_files = [item for sublist in image_files for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        return image_files,Tensor(labels)

    def _get_count(self):
        count = 0
        dirs = os.listdir(FaceDataset.GREYSCALE_DATA)
        for dir in dirs:
            count += len(os.listdir(os.path.join(FaceDataset.GREYSCALE_DATA,dir)))
        return count
    
    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        file_path = self.image_files[index]
        label = self.labels[index]
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image,label

class FaceDetectionDataset:
    
    def __init__(self,truth_label_path:str) -> None:
        self.truth_label_path = truth_label_path
        self.image_paths, self.bboxes = self.get_image_paths(1,100)
        

    def get_image_paths(self,number_of_faces_in_image,number_of_images_retrieved):
        image_path = []
        boxes = []
        with open(self.truth_label_path,'r') as f:
            lines = f.readlines()
        for idx,line in enumerate(lines):
            if re.match(f'^{number_of_faces_in_image}\n',line):
                image_path.append(lines[idx-1].strip())
                box = lines[idx+1].strip().split(' ')[0:4]
                box = [int(x) for x in box]
                boxes.append(box)
                if len(image_path) >= number_of_images_retrieved:
                    return image_path, boxes
        return image_path, boxes

    def stream_data(self):
        for image_path, bbox in zip(self.image_paths,self.bboxes):
            img = Image.open(os.path.join('data/WIDER_val/images',image_path))
            yield img,bbox

if __name__ == '__main__':
    process = FaceDataset()
    process = FaceDetectionDataset('data/wider_face_split/wider_face_val_bbx_gt.txt')
