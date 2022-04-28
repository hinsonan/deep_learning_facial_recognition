'''
description:This file will tie the models together and generate an 
output of an image with a bounding box and emotional label
'''
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
from pathlib import Path
from detector_pipeline import DetectorPipeline
from torchvision import transforms

if __name__ == '__main__':
    detect = DetectorPipeline(backbone_model='MTCNN')
    # load the best model
    model = torch.load(Path('models/vgg.pt'),map_location='cpu')
    model.eval()
    # load image
    test_image = Path('data/WIDER_val/images/52--Photographers/52_Photographers_photographertakingphoto_52_807.jpg')
    img = Image.open(test_image)
    boxes = detect.detector.detect(np.array(img))

    # draw box
    for box in boxes:
        # clip the bounding box dimensions to the image size
        box = np.clip(box,0,img.size[0]-2)
        box = [int(x) for x in box]
        img_draw = ImageDraw.Draw(img)
        img_draw.rectangle(box,outline='blue',width=5)
    img.show()
    im1 = img.crop(box)
    im1.show()

    # resize and greyscale image
    im1 = ImageOps.grayscale(im1)
    transform = transforms.Compose([
            transforms.Resize(size=(128,128)),
            transforms.ToTensor(),
            # note that this may not be the most accurate way to normalize
            transforms.Normalize(mean=[0.5],std=[0.5])
        ])
    im1 : torch.Tensor = transform(im1)
    im1 = im1.unsqueeze(0)
    out = model(im1)
    label = np.argmax(out.detach().numpy())
    print(label)

