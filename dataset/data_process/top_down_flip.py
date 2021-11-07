import cv2
from PIL import Image
import random
import os

path = '/home/p920/cf/data2/train/'
files = os.listdir(path)
target = '/home/p920/cf/data3/'

for file in files:
    if 'Annotation' not in file:
        if random.random() < 0.5:
            name = file.split('.')[0]
            img = Image.open(os.path.join(path,file))
            img_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_flip.save(target+name+'_flip.png')

            label = Image.open(path+name+'_Annotation.png')
            label_flip = label.transpose(Image.FLIP_TOP_BOTTOM)
            label_flip.save(target + name + '_flip_Annotation.png')




