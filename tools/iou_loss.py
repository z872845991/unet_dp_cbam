import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        return _iou(pred, target, self.size_average)




if __name__ == "__main__":
    a = torch.zeros((2,1,512,512))
    b = torch.ones((2,1,512,512))
    criterion = IOU()

    import torch.nn as nn
    criterion1 = nn.BCEWithLogitsLoss()
    loss1 = criterion(a,b)
    loss2 = criterion1(a,b)
    loss3 = criterion(F.sigmoid(a),b)

    print(loss1)
    print(loss2)
    print(loss3)