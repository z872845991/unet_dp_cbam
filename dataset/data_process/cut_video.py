import cv2
import numpy as np
import os

# 把视频截取成图片
path = 'E:\\workspace\\dataset\\fetal_video\\'
target = 'E:\\workspace\\dataset\\video_test\\'
files = os.listdir(path)
for file in files:
    print(file)
    if file.split('.')[-1]=='avi':
        res = cv2.VideoCapture(os.path.join(path,file))
        retrival = res.isOpened()
        # success, frame = cv2.VideoCapture.read(res)
        # print(frame.shape)
        num=0
        while retrival:
            success, frame = res.read()
            num+=1
            if success:
                cv2.imwrite('E:\\workspace\\dataset\\video_test\\%s_%d.jpg'%(file,num),frame)
            else:
                break
        res.release()

    else:
        continue
