from typing import List
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcess():

    GREYSCALE_DATA = 'data/greyscale_faces'
    RGB_DATA = 'data/rgb_faces'

    def __init__(self) -> None:
        self.label_encoding = LabelEncoder()

    def get_all_data(self) -> np.array:
        data = []
        labels = []
        for _,dirs,_ in os.walk(DataProcess.GREYSCALE_DATA):
            label_encode = self.label_encoding.fit_transform(dirs)
            for idx,dir in enumerate(dirs): 
                for _,_,files in os.walk(f'{DataProcess.GREYSCALE_DATA}/{dir}'):
                    for file in files:
                        image = cv2.imread(f'{DataProcess.GREYSCALE_DATA}/{dir}/{file}', cv2.IMREAD_GRAYSCALE)
                        data.append(image)
                        labels.append(label_encode[idx])
        return np.asarray(data), np.asarray(labels)
    
    def split_data(self,X,y) -> list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=12)

        return X_train,X_val,X_test,y_train,y_val,y_test

    def normalize(self,images: np.array) -> np.array:
        norm_img = np.zeros(images.shape)
        normalized_image = cv2.normalize(images,  norm_img, 0.0, 1.0, cv2.NORM_MINMAX,cv2.CV_32FC1)
        return normalized_image
        

if __name__ == '__main__':
    process = DataProcess()
    data,labels = process.get_all_data()
    norm_data = process.normalize(data)
    print(norm_data)
