import cv2

import numpy as np

t = cv2.imread("img_repo/models/test_label.png", cv2.IMREAD_GRAYSCALE)
O = cv2.equalizeHist(t)
cv2.imshow("a", O)
cv2.imwrite("result2.png", O)
print("down")
cv2.waitKey()
