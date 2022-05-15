import cv2
import numpy as np
import dlib
import utils
out = "1.mp4"
fps = 20
fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # 支持jpg
videoWriter = cv2.VideoWriter(out, fourcc, fps, (640, 480))
cap = cv2.
while True:

for im_name in range(len(im_names) - 2):
    string = img_root + 'frame' + str(im_name) + '.jpg'
    print(string)
    frame = cv2.imread(string)
    frame = cv2.resize(frame, (640, 480))
    videoWriter.write(frame)

