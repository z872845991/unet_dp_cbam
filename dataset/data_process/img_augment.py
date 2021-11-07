import cv2
import os
import numpy as np

# 本代码是手动的对分割的数据集做一个增强操作


def prepare(file, path,target):
    if "Annotation" in file:
        name = file.split('_')[0]
        ten_crop(file, path, name, target,"label")
        # make_noise(file, path, name, target,"label")
    else:
        name = file.split('.')[0]
        ten_crop(file,path,name,target,None)

# 五角裁剪
def five_crop(file,path,name,target,label=None):
    img = cv2.imread(os.path.join(path,file))
    height,width,channel = img.shape
    midH,midW = height//2,width//2
    img1 = img[:512,:512]
    img2 = img[height-512:,:512]
    img3 = img[:512,width-512:]
    img4 = img[height-512:,width-512:]
    img5 = img[midH-256:midH+256,midW-256:midW+256]
    if label==None:
        cv2.imwrite(target + name+'_lu.png',img1)
        cv2.imwrite(target + name+'_ld.png',img2)
        cv2.imwrite(target + name+'_ru.png',img3)
        cv2.imwrite(target + name+'_rd.png',img4)
        cv2.imwrite(target + name+'_mid.png',img5)
    else:
        cv2.imwrite(target + name + '_lu_Annotation.png', img1)
        cv2.imwrite(target + name + '_ld_Annotation.png', img2)
        cv2.imwrite(target + name + '_ru_Annotation.png', img3)
        cv2.imwrite(target + name + '_rd_Annotation.png', img4)
        cv2.imwrite(target + name + '_mid_Annotation.png', img5)


# 十角裁剪
def ten_crop(file,path,name,target,label=None):
    img = cv2.imread(os.path.join(path, file))
    height, width, channel = img.shape
    midH, midW = height // 2, width // 2
    # 第一行
    img11 = img[:512, :512]
    img12 = img[:512, midW-256:midW+256]
    img13 = img[:512, width-512:]

    img21 = img[midH-256:midH+256, :512]
    img22 = img[midH-256:midH+256, midW-256:midW+256]
    img23 = img[midH-256:midH+256, width-512:]

    img31 = img[height-512:, :512]
    img32 = img[height-512:, midW-256:midW+256]
    img33 = img[height-512:, width-512:]

    # img_center = img[midH - 256:midH + 256, midW - 256:midW + 256]

    if label == None:
        cv2.imwrite(target + name + '_11.png', img11)
        cv2.imwrite(target + name + '_12.png', img12)
        cv2.imwrite(target + name + '_13.png', img13)
        cv2.imwrite(target + name + '_21.png', img21)
        cv2.imwrite(target + name + '_22.png', img22)
        cv2.imwrite(target + name + '_23.png', img23)
        cv2.imwrite(target + name + '_31.png', img31)
        cv2.imwrite(target + name + '_32.png', img32)
        cv2.imwrite(target + name + '_33.png', img33)
        # cv2.imwrite(target + name + '_center.png', img_center)


    else:
        cv2.imwrite(target + name + '_11_Annotation.png', img11)
        cv2.imwrite(target + name + '_12_Annotation.png', img12)
        cv2.imwrite(target + name + '_13_Annotation.png', img13)
        cv2.imwrite(target + name + '_21_Annotation.png', img21)
        cv2.imwrite(target + name + '_22_Annotation.png', img22)
        cv2.imwrite(target + name + '_23_Annotation.png', img23)
        cv2.imwrite(target + name + '_31_Annotation.png', img31)
        cv2.imwrite(target + name + '_32_Annotation.png', img32)
        cv2.imwrite(target + name + '_33_Annotation.png', img33)
        # cv2.imwrite(target + name + '_center_Annotation.png', img_center)

def make_noise(file,path,name,target,label=None):
    if label != None:
        image  = cv2.imread('3.png')  # a black objects on white image is better

        cv2.rectangle(image, (300, 300), (500, 500), (0, 255, 0), 1)
        cv2.imshow('1',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    path="/home/p920/cf/data/train/"
    target = "/home/p920/cf/data/aug_train/"
    files = os.listdir(path)
    for file in files:
        prepare(file,path,target)

if __name__=="__main__":
    main()