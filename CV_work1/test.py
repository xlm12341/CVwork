import cv2
import numpy as np
import dlib
import utils

cap = cv2.imread("img_repo/01_background.jpg")

img2 = cv2.resize(cap,(640, 480),interpolation=cv2.INTER_LINEAR)
cv2.imshow("s", img2)
cv2.imwrite("../CV_work2/img_repo/backgrounds/01_background.jpg", img2)
print("done")
