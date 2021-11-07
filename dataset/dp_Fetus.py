from torch.utils.data import Dataset
import PIL.Image as Image
import os
import re
import cv2
import numpy as np
from torchvision.transforms import transforms

def load_file_name(path):
    files = os.listdir(path)
    imgs = []

    for file in files:
        # if not re.search('\Snnotation',file) :
        if not 'Annotation' in file:
            label = file.split('.')[0]+'_Annotation.png'
            imgs.append((file,label))

    return imgs

def make_dataset(path):
    return load_file_name(path)

# print(make_dataset('./data'))

class FetusDataset(Dataset):
    def __init__(self,path,mode='train',transform=None, target_transform=None):
        super(FetusDataset,self).__init__()
        self.path = path
        self.mode = mode
        self.imgs = make_dataset(path)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        if self.mode=='train':
            labellist=[]
            img,label = self.imgs[index]
            img = Image.open(os.path.join(self.path, img)).convert("RGB")
            label = Image.open(os.path.join(self.path, label)).convert('L').resize((512,512))
            label1 = label.resize((64,64))
            label2 = label.resize((128,128))
            label3 = label.resize((256,256))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)
                label1 = self.target_transform(label1)
                label2 = self.target_transform(label2)
                label3 = self.target_transform(label3)
                labellist.append(label1)
                labellist.append(label2)
                labellist.append(label3)
                labellist.append(label)
            labellist = np.array(labellist)
            return img,labellist

        else:
            img, label = self.imgs[index]

            img = Image.open(os.path.join(self.path, img))
            if self.transform is not None:
                img = self.transform(img)
            return img





    def __len__(self):
        return len(self.imgs)

