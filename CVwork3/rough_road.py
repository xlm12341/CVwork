import cv2
import numpy as np

img = cv2.imread("4.jpg")
img02 = cv2.imread("04.png")
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
low_green = np.array([56, 146, 0])
high_green = np.array([89, 241, 255])
mask = cv2.inRange(hsv_img, low_green, high_green)

res = cv2.bitwise_and(img02, img02, mask=mask)


cv2.imshow("res", res)
cv2.imwrite("02res.png", res)

img = cv2.imread("02res.png")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(imggray,127, 255,0)
cv2.imshow("binary", binary)
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))
dst = cv2.drawContours(img, contours, -1, (0,0,255),1)
cv2.imshow("dst", dst)
cv2.imwrite("dst2.png", dst)
cv2.waitKey(0)