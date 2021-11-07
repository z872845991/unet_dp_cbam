import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


if __name__ == "__main__":
    a = torch.randn((2,3,512,512)).cuda()
    from model.unet import Unet
    model = Unet(3,1).cuda()
    output = model(a)

    label = torch.zeros((2,1,512,512)).cuda()
    criterion = FocalLoss()
    loss1 = criterion(output,label)

    loss2 = F.binary_cross_entropy_with_logits(output,label)

    print(loss1)
    print(loss2)
