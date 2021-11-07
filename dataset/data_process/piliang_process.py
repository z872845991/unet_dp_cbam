# 批量处理GT图像，阈值分割

import cv2
import os

path = "F:\\dataset\\old_label\\"
files = os.listdir(path)
print(len(files))
target = "F:\\dataset\\new_label\\"
i=0
for file in files:
    if 'Annotation' in file:
        img = cv2.imread(os.path.join(path,file),0)
        i+=1
        ret,dst = cv2.threshold(img,8,255,cv2.THRESH_BINARY)
        # newname = "E:\\workspace\\dataset\\cerv_1\\newlabel\\" + str(i) + "_Annotation.png"
        # print(newname)
        cv2.imwrite(os.path.join(target,file),dst)

print(i)