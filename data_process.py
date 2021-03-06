import os
import re
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch import Tensor

class FaceDataset(Dataset):

    GREYSCALE_DATA = f'data{os.sep}greyscale_faces'
    AUGMENT_DATA = f'data{os.sep}augmentation'

    def __init__(self,transform=None,use_augment=False) -> None:
        self.use_augment = use_augment
        self.label_encoding = self._get_label_encodings()
        self.transform = transform
        self.image_files, self.labels = self._get_all_image_paths_and_labels()
        self.data_count = self._get_count()
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
        if self.use_augment:
            dirs = os.listdir(FaceDataset.AUGMENT_DATA)
            for dir in dirs:
                file_list = os.listdir(os.path.join(FaceDataset.AUGMENT_DATA,dir))
                for idx,f_string in enumerate(file_list):
                    file_list[idx] = os.path.join(FaceDataset.AUGMENT_DATA,dir,f_string)
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
        if self.use_augment:
            dirs = os.listdir(FaceDataset.AUGMENT_DATA)
            for dir in dirs:
                count += len(os.listdir(os.path.join(FaceDataset.AUGMENT_DATA,dir)))
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
        self.count = 500
        self.image_paths, self.bboxes = self.get_image_paths(1,self.count)
        

    def get_image_paths(self,number_of_faces_in_image,number_of_images_retrieved):
        image_path = []
        boxes = []
        with open(self.truth_label_path,'r') as f:
            lines = f.readlines()
        for idx,line in enumerate(lines):
            if re.match(f'^{number_of_faces_in_image}\n',line):
                box = lines[idx+1].strip().split(' ')[0:4]
                box = [int(x) for x in box]
                # check if the bounding box is all 0s and dont use this image
                if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0: 
                    continue
                boxes.append(box)
                image_path.append(lines[idx-1].strip())
                if len(image_path) >= number_of_images_retrieved:
                    return image_path, boxes
        return image_path, boxes

    def stream_data(self):
        for image_path, bbox in zip(self.image_paths,self.bboxes):
            img = Image.open(os.path.join(f'data{os.sep}WIDER_val{os.sep}images',image_path))
            yield img,bbox

if __name__ == '__main__':
    process = FaceDataset()
    process = FaceDetectionDataset(f'data{os.sep}wider_face_split{os.sep}wider_face_val_bbx_gt.txt')
