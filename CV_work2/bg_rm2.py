from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2

seg = SelfiSegmentation()
img = cv2.imread("Project02-example/img/05/try_on5.png")
bg = cv2.imread("Project02-example/img/05/05_background.jpg")
bg2 = cv2.resize(bg,(img.shape[1], img.shape[0]),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
img2 = seg.removeBG(img,bg2,threshold=0.5)
cv2.imshow("s", img2)
cv2.imwrite("Project02-example/img/05/try_on5_bg.png", img2)
cv2.waitKey()
