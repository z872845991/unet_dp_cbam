# 本代码是用来找到直到最后一个epoch，iou还是很低的图片
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset.Fetus import FetusDataset
# from model.seunet import Unet
# from unet import Unet
# from archs import NestedUNet
# from ince_unet import Unet
from eca_unet import Unet
from metrics import dice_coef,iou_score
from tools.utils import AverageMeter
import datetime

# import visdom


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

x_transforms = transforms.Compose([
    transforms.Resize((512,800)),
    # transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512, 800)),
    # transforms.CenterCrop(512),
    transforms.ToTensor()
   ])

#参数解析

def train_model(model, criterion, optimizer, dataload, num_epochs):
    # 这个是用来找到miou最好的一次epoch
    bigiou = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        avgmeter1 = AverageMeter()
        avgmeter2 = AverageMeter()
        step = 0
        for x,y in dataload:
            step += 1
            inputs = x.cuda()
            labels = y.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算loss和iou指标
            loss = criterion(outputs, labels)
            iou = iou_score(outputs,labels)
            dice = dice_coef(outputs,labels)

            avgmeter1.update(iou,args.batch_size)
            avgmeter2.update(dice,args.batch_size)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            print("%d/%d,train_loss:%0.3f miou:%.3f dice:%.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(), avgmeter1.avg, avgmeter2.avg))
        # 这里的最大最小iou是训练集中分的最好最差的数据
        print("epoch %d loss:%0.3f miou:%.3f maxiou:%.3f miniou:%.3f  mdice:%.3f maxdice:%.3f mindice:%.3f" % (epoch, epoch_loss/step, avgmeter1.avg,avgmeter1.max,avgmeter1.min, avgmeter2.avg,avgmeter2.max,avgmeter2.min))
        with open('./result/train_eca_unet.txt','a+') as file:
            file.write("epoch %d loss:%0.3f miou:%.3f maxiou:%.3f miniou:%.3f  mdice:%.3f maxdice:%.3f mindice:%.3f" % (epoch, epoch_loss/step, avgmeter1.avg,avgmeter1.max,avgmeter1.min, avgmeter2.avg,avgmeter2.max,avgmeter2.min) + '\n')
        if epoch==num_epochs-1 or avgmeter1.avg>bigiou :
            bigiou = avgmeter1.avg
            torch.save(model.state_dict(), './checkpoints/weights_eca_unet_%d.pth' % epoch)


        test_model(model)

#训练模型
def train(args):
    model = Unet(3,1)
    model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    # fetus_dataset = FetusDataset("E:\\workspace\\pytorch\\simple_u_net_v2\\dataset\\data\\",transform=x_transforms,target_transform=y_transforms)
    # fetus_dataset = FetusDataset("E:\\train\\", transform=x_transforms,target_transform=y_transforms)
    fetus_dataset = FetusDataset("E:\\workspace\\dataset\\crev_segmentation\\fuse-all\\train\\", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(fetus_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last=False)
    train_model(model, criterion, optimizer, dataloaders,args.epoches)


def test_model(model):
    test_dataset = FetusDataset("E:\\workspace\\dataset\\crev_segmentation\\fuse-all\\val\\", mode='train',transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(test_dataset, batch_size=1)
    avgmeter3 = AverageMeter()
    avgmeter4 = AverageMeter()
    step = 0
    for x, y in dataloaders:
        step += 1
        inputs = x.cuda()
        labels = y.cuda()
        outputs = model(inputs)

        # 计算loss和iou指标
        iou1 = iou_score(outputs, labels)
        dice1 = dice_coef(outputs, labels)

        avgmeter3.update(iou1, args.batch_size)
        avgmeter4.update(dice1, args.batch_size)

        # print("miou:%.3f dice:%.3f" % (avgmeter3.avg, avgmeter4.avg))
    print("Val miou:%.3f maxiou:%.3f miniou:%.3f  mdice:%.3f maxdice:%.3f mindice:%.3f" % (
         avgmeter3.avg, avgmeter3.max, avgmeter3.min, avgmeter4.avg, avgmeter4.max,avgmeter4.min))
    with open('./result/val_eca_unet.txt', 'a+') as file:
        file.write(" miou:%.3f maxiou:%.3f miniou:%.3f  mdice:%.3f maxdice:%.3f mindice:%.3f" % (
        avgmeter3.avg, avgmeter3.max, avgmeter3.min, avgmeter4.avg, avgmeter4.max,avgmeter4.min) + '\n')


#显示模型的输出结果
# def test():
#     model = Unet(3,1)
#     # model = NestedUNet(1)
#     model.load_state_dict(torch.load("./checkpoints/weights_Ecaunet_80.pth"),strict=False)
#     model = model.cuda()
#     # liver_dataset = FetusDataset("E:\\workspace\\dataset\\test/clear", transform=x_transforms)
#     # liver_dataset = FetusDataset("E:\\workspace\\dataset\\test/no", transform=x_transforms)
#     # fetus_dataset = FetusDataset("E:\\workspace\\dataset\\test\\5\\",mode='test', transform=x_transforms)
#     fetus_dataset = FetusDataset("E:\\workspace\\dataset\\crev_segmentation\\fuse-all\\val\\",mode='test', transform=x_transforms)
#     dataloaders = DataLoader(fetus_dataset, batch_size=1)
#     model.eval()
#     import matplotlib.pyplot as plt
#     import matplotlib
#     plt.ion()
#     i=0
#     avgmeter = AverageMeter()
#     with torch.no_grad():
#         for input in dataloaders:
#             input = input.cuda()
#             y=model(input)
#
#             # 计算评价指标
#             # iou = iou_score(y,gt)
#             # avgmeter.update(iou,1)
#
#             # 保存预测的图片
#             y = y.cpu()
#             img_y=torch.squeeze(y).numpy()
#             matplotlib.image.imsave('E:\\workspace\\dataset\\test\\1\\eca80_%d.png'%i,img_y)
#             i+=1

        # print("miou:%.4f" %avgmeter.avg)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--epoches", type=int, default=81)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # viz = visdom.Visdom(env='cer_seg')
    start = datetime.datetime.now()
    train(args)
    # test()
    end = datetime.datetime.now()
    print(end-start)
