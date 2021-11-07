import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


# 准确率 ACC
def get_accuracy(output,target):
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()

    # output处理成0-1
    threshold = 0.5
    tmp_out = (output >= threshold)
    tmp_out = tmp_out.astype(float)

    # 预测和label相同时
    index = (tmp_out == target)
    index = index.astype(float)
    tmp_sum = np.sum(index)

    # 特征像素大小
    target_num = target.size

    return tmp_sum/target_num

# 特异度 TNR
def get_specificity(output,target):
    output = torch.sigmoid(output).data.cpu().numpy()
    # output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    # 将输出的值四舍五入到0,1之中
    threshold = 0.5
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)

    # 输出和label += 0，则两个都为0的部分就为TN
    tn_out = thre_out + target
    tn_out = (tn_out == 0)
    TN = np.sum(tn_out)

    # 真实为N的部分为Gt_neg
    Gt_neg = (target == 0)
    Gt_neg = np.sum(Gt_neg)

    return TN / (Gt_neg+1e-6)

# 精确度 PPV
def get_precision(output,target):
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()

    # 将输出的值四舍五入到0,1之中
    threshold = 0.5
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)

    # 输出和label += 2，则两个都为1的部分就为TP
    tp_out = thre_out + target
    tp_out = (tp_out==2)
    TP = np.sum(tp_out)

    # 预测为真的部分为Ppe_pos
    Pre_pos = (thre_out==1)
    Pre_pos = np.sum(Pre_pos)

    return TP/(Pre_pos+1e-6)

# 灵敏度 TPR
def get_recall(output,target):
    output = torch.sigmoid(output).data.cpu().numpy()
    # output = output.data.cpu().numpy()
    target = target.data.cpu().numpy()

    # output = torch.sigmoid(output).data.cpu().numpy()
    # target = target.data.cpu().numpy()

    # 将输出的值四舍五入到0,1之中
    threshold = 0.5
    thre_out = (output >= threshold)
    thre_out = thre_out.astype(float)

    # 输出和label += 2，则两个都为1的部分就为TP
    tp_out = thre_out + target
    tp_out = (tp_out == 2)
    TP = np.sum(tp_out)

    # label为真的部分为Gt_pos
    Gt_pos = (target == 1)
    Gt_pos = np.sum(Gt_pos)

    return TP /(Gt_pos+1e-6)

# F1
def get_F1(output,target):
    recall = get_recall(output,target)
    precision = get_precision(output,target)
    F1 = 2*recall*precision/(recall+precision+1e-6)

    return F1



if __name__ == "__main__":
    # target = np.array([[[[0,1,0],
    #                    [1,1,1],
    #                    [0,1,0]]],
    #                    [[[1, 1, 1],
    #                      [1, 1, 1],
    #                      [1, 1, 1]]]
    #                    ])
    # target = torch.from_numpy(target)
    # output = np.array([[[[0,0,0],
    #                    [1,1,1],
    #                    [0,0,0]]],
    #                    [[[0, 0, 0],
    #                      [0, 1, 0],
    #                      [0, 0, 0]]]
    #                    ])
    # output = torch.from_numpy(output)
    # print(target.shape)
    # print(output.shape)
    #
    # index = (target==output)
    # print(index)
    #
    # a = torch.sum(index)
    # b = output.nelement()
    # print(a)
    # print(b)
    # print(float(a.numpy()/b))

    output = np.array([[[[0.4, 0.7, 0.3],
                       [1, 0.9, 0.8],
                       [0.3, 0.5, 0.1]]],
                       [[[0.5, 1, 1],
                         [1, 0.6, 1],
                         [1, 1, 0.7]]]
                       ])
    output = torch.from_numpy(output)
    target = np.array([[[[0, 0, 0],
                       [1, 1, 1],
                       [0, 0, 0]]],
                       [[[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]]
                       ])
    target = torch.from_numpy(target)

    # pa = piexl_accuracy(output,target)
    # print(pa)
    # pre = piexl_precision(output,target)
    # print(pre)

    # pre = get_recall(output,target)
    # print(pre)

    tnr = get_specificity(output,target)
    print(tnr)

    # threshold = 0.5
    # tmp = output>=threshold
    # print(tmp)
    # target = tmp.numpy().astype(float)
    # target = torch.from_numpy(target)
    # index = (target==output)
    # print(index)
    #
    # a = torch.sum(index)
    # b = output.nelement()
    # print(a)
    # print(b)
    # print(float(a.numpy()/b))