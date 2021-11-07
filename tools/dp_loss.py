import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from model.dp_unet import Unet
import numpy as np




class Dp_loss(nn.Module):
    def __init__(self):
        super(Dp_loss,self).__init__()
    #     将标签文件降采样到64,128,256,512,512
        self.BCEloss = nn.BCEWithLogitsLoss()

    def forward(self,input,target):
        l1 = self.BCEloss(input[0], target[0])
        l2 = self.BCEloss(input[1], target[1])
        l3 = self.BCEloss(input[2], target[2])

        l4 = self.BCEloss(input[3], target[3])
        loss = 0.3 * l1 + 0.3 * l2 + 0.3 * l3 + l4
        return loss


    def _downsample(self,label,size):
        target = label.clone()
        batch = target.shape[0]
        channel = target.shape[1]
        height = target.shape[2]
        contain = []
        for m in range(batch):
            contain.append(np.zeros((1,channel,size[0],size[1])))

        j=0
        for i in target:
            i = i.numpy()
            i = np.transpose(i,(1,2,0))
            i = cv2.resize(i,size)
            # 灰度图resize后少了通道维度，另外再加个batch维度
            i = torch.tensor(i).unsqueeze(0).unsqueeze(0)
            contain[j]=i
            j+=1

        #
        output = torch.cat((contain[0],contain[1]),0)
        return output



if __name__=="__main__":
    input = torch.randn((2,3,512,512))
    label = torch.randn((2,1,512,512))
    # print(type(input))
    model = Unet(3,1)
    output = model(input)
    criterion = Dp_loss()
    loss = criterion(output,label)
    print(loss)



    """
        def forward(self,input, target):
            l1 = self.BCEloss(input[0],self._downsample(target,(64,64)))
            l2 = self.BCEloss(input[1],self._downsample(target,(128,128)))
            l3 = self.BCEloss(input[2],self._downsample(target,(256,256)))

            l4 = self.BCEloss(input[3],target)
            loss = 0.3*l1+0.3*l2+0.3*l3+l4
            return loss
    """


    """
    a = torch.randn((2, 3, 512, 512))
    # a = np.array((512,512,3),dtype=np.uint8)
    a = a.numpy()
    tmp1 = np.zeros((3, 256, 256))
    tmp2 = np.zeros((3, 256, 256))
    j = 0
    for i in a:
        i = np.transpose(i, (1, 2, 0))
        b = _downsample(i, (256, 256))
        if j == 0:
            tmp1 = np.transpose(b, (2, 0, 1))
            tmp1 = torch.tensor(tmp1)
            tmp1 = tmp1.unsqueeze(dim=0)
            j += 1
        else:
            tmp2 = np.transpose(b, (2, 0, 1))
            tmp2 = torch.tensor(tmp2)
            tmp2 = tmp2.unsqueeze(dim=0)

    b = torch.cat((tmp1, tmp2), dim=0)
    print(b.shape)
    """