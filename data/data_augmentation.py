import os
import numpy as np
from PIL import Image, ImageFilter,ImageEnhance
from torchvision import transforms
from pathlib import Path
if __name__ == '__main__':
    dirs = os.listdir(Path('data/greyscale_faces'))

    for dir in dirs:
        files = os.listdir(Path(f'data/greyscale_faces/{dir}'))
        for file in files:
            img = Image.open(Path(f'data/greyscale_faces/{dir}/{file}'))
            # flip image
            flipped_img = np.fliplr(np.array(img))
            # apply blur
            blur_img = img.filter(ImageFilter.GaussianBlur(2))
            contrast_img = ImageEnhance.Contrast(img).enhance(1.5)
            if not os.path.isdir(Path(f'data/augmentation/{dir}')):
                os.mkdir(Path(f'data/augmentation/{dir}'))
            # write img to file
            flipped_img = Image.fromarray(flipped_img)
            flipped_img.save(f'data/augmentation/{dir}/{file}_augment_flip.png')
            blur_img.save(f'data/augmentation/{dir}/{file}_augment_blur.png')
            contrast_img.save(f'data/augmentation/{dir}/{file}_augment_contrast.png')
