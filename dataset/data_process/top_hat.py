import cv2
import numpy as np

img = cv2.imread("2.png",0)
kernel = np.ones((5,5),np.uint8)
img1 = cv2.erode(img,kernel=kernel,iterations=2)
cv2.imshow("source",img)
cv2.imshow('1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()