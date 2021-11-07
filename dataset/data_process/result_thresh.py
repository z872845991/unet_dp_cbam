import cv2
import os


img = cv2.imread('eac_0.png',0)

ret,dst = cv2.threshold(img,8,255,cv2.THRESH_BINARY)

cv2.imwrite('thresh_eca_0.png',dst)

# cv2.imshow('dst',dst)
#
#
# cv2.waitKey()
# cv2.destroyAllWindows()