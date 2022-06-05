import cv2

cap = cv2.VideoCapture("./06.mp4")
FPS = cap.get(5)
print(FPS)

c = 1
while True:
    ret, frame = cap.read()
    if ret:
        if 80 < c < 90:
            cv2.imwrite("./cap_img" + str(c) + '.jpg', frame)
    c = c + 1

print("done")
cap.release()
exit(0)