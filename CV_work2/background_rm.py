from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2
import cvzone
import mediapipe
import numpy
import os
cap = cv2.VideoCapture(0)
seg = SelfiSegmentation()
fps_reader = cvzone.FPS()
bk_img = cv2.imread("img_repo/backgrounds/01_background2.jpg")
dir_list = os.listdir("img_repo")
bk_img_list = []

for name in dir_list:
    img_tmp = cv2.imread(f"img_repo/{name}")
    print(img_tmp.shape)
    bk_img_list.append(img_tmp)


bk_img_index = 0
while True:
    success, img = cap.read()
    img_p = seg.removeBG(img, bk_img_list[bk_img_index], threshold=0.4)

    stacked_img = cvzone.stackImages([img, img_p], 2, 1)
    _, stacked_img = fps_reader.update(stacked_img)
    cv2.imshow("img", stacked_img)

    key = cv2.waitKey(1)
    if key == ord('a'):
        if bk_img_index > 0:
            bk_img_index = bk_img_index -1
    elif key == ord('d'):
        if bk_img_index < len(bk_img_list) - 1:
            bk_img_index = bk_img_index + 1
    elif key == ord('q'):
        break

exit(0)
