import cv2
import os

# 对训练集和数据集，去除个人信息

path = "/home/p920/cf/source_data/test/"
"""
train:[(852, 1136), (852, 1604), (576, 768)]
val:
"""
target = "/home/p920/cf/data/test/"
files = os.listdir(path)

list = []
for file in files:
    if os.path.isdir(file):
        continue
    if "Annotation" in file:
        continue

    img_name = file.split('.')[0]
    label_name = img_name + "_Annotation.png"

    img = cv2.imread(os.path.join(path,file),1)
    label = cv2.imread(os.path.join(path,label_name),0)
    # print(img.shape)
    if img is None:
        continue
    height = img.shape[0]
    width = img.shape[1]
    # 241.png  112   105
    img_croped = img[70:-20,70:-90]
    label_croped = label[70:-20,70:-90]

    cv2.imwrite(target+file,img_croped)
    cv2.imwrite(target+label_name,label_croped)





# print(list)
