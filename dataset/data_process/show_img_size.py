import os
import cv2

path = "F:\\dataset\\cerv_1\\train\\"

files = os.listdir(path)

for file in files:
    img = cv2.imread(path+file)
    print(img.shape)
