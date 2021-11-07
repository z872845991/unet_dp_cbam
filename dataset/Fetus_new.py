from torch.utils.data import Dataset
import PIL.Image as Image
import os
import re
import cv2
import numpy as np
import dataset.custom_transforms as tr
from torchvision import transforms


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
            img, label = self.imgs[index]
            name = img
            _img = Image.open(os.path.join(self.path, img)).convert("RGB")
            # img = Image.open(os.path.join(self.path,img)).crop((200,200,1000,712))
            # label = Image.open(os.path.join(self.path,label)).convert('L').crop((200,200,1000,712))
            _target = Image.open(os.path.join(self.path, label)).convert('L')
            sample = {'image': _img, 'label': _target}
            sample = self.transform_tr(sample)

            _img = sample['image']
            _target = sample['label']
            if self.transform is not None:
                _img = self.transform(_img)
            if self.target_transform is not None:
                _target = self.target_transform(_target)



            return _img, _target, name


        elif self.mode=='val':
            img,label = self.imgs[index]
            name = img
            img = Image.open(os.path.join(self.path, img)).convert("RGB")
            # img = Image.open(os.path.join(self.path,img)).crop((200,200,1000,712))
            # label = Image.open(os.path.join(self.path,label)).convert('L').crop((200,200,1000,712))
            label = Image.open(os.path.join(self.path, label)).convert('L')
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)


            return img,label,name

        else:
            img, label = self.imgs[index]

            img = Image.open(os.path.join(self.path, img))
            if self.transform is not None:
                img = self.transform(img)
            return img



    def transform_tr(self, sample):       #对训练集处理
        composed_transforms = transforms.Compose([
            tr.RandomRotate(45),
            # tr.FixedResize(512),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=512, crop_size=512),
            tr.RandomGaussianBlur()
            # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # tr.ToTensor()
        ])
        return composed_transforms(sample)



    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    x_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.RandomErasing(),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
    ])
    from torch.utils.data import DataLoader
    fetus_dataset = FetusDataset("F:\\dataset\\train\\", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(fetus_dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True)

    for x,y,_ in dataloaders:
        print(x.shape)
        print(y.shape)
