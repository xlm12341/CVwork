import cv2
import numpy as np
def imgBrightness(img1, c, b): 
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst
img = cv2.imread('Project02-example/img/03/000001_0 (1).png')
#第二个参数调节亮度，越大越亮，越小越暗
#变暗：rst = contrast_img(img, 0.5, 3)
rst = imgBrightness(img, 6, 3)
cv2.imwrite('1.png', rst)
