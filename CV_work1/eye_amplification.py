import cv2
import dlib
import math
import numpy as np

webcam = False


def empty(a):
    pass


# def bilinear_insert(image, ux, uy):
#     x1 = int(ux)
#     x2 = x1 + 1
#     y1 = int(uy)
#     y2 = y1 + 1
#
#     part1 = image[y1, x1].astype(np.float64) * (float(x2) - ux) * (float(y2) - uy)
#     part2 = image[y1, x2].astype(np.float64) * (ux - float(x1)) * (float(y2) - uy)
#     part3 = image[y2, x1].astype(np.float64) * (float(x2) - ux) * (uy - float(y1))
#     part4 = image[y2, x2].astype(np.float64) * (ux - float(x1)) * (uy - float(y1))
#
#     insert_val = part1 + part2 + part3 + part4
#     return insert_val


def bigger_eye(image, eye_landmark, down, radius, strength=20):
    # height, width, channels = image.shape
    # for i in range(height):
    #     for j in range(width):
    #         if abs(eye_landmark.x - j) > radius or abs(eye_landmark.y - i) > radius:
    #             continue
    #         distance = (eye_landmark.x - j) * (eye_landmark.x - j) + (eye_landmark.y - i) * (eye_landmark.y - i)
    #         if distance < radius * radius:
    #             norm = math.sqrt(distance) / radius
    #             ratio = 1 - (norm - 1) * (norm - 1) * 0.5
    #             ux = eye_landmark.x + ratio * (j - eye_landmark.x)
    #             uy = eye_landmark.y + ratio * (i - eye_landmark.y)
    #             val = bilinear_insert(image, ux, uy)
    #             image[i, j] = val
    # return img

    height, width, channels = image.shape
    mapx = np.vstack([np.arange(width).astype(np.float32).reshape(1, -1)] * height)
    mapy = np.hstack([np.arange(height).astype(np.float32).reshape(-1, 1)] * width)
    offset_x = mapx - eye_landmark.x
    offset_y = mapy - eye_landmark.y
    XY = offset_x * offset_x + offset_y * offset_y
    # print(np.where(XY < 10))
    pow_radius = radius * radius
    # print(pow_radius)
    scale_factor = 1 - XY / pow_radius
    # print(np.where(scale_factor>0))
    scale_factor = 1 - strength / 100 * scale_factor
    # print(scale_factor)
    UX = offset_x * scale_factor + eye_landmark.x
    UY = offset_y * scale_factor + eye_landmark.y
    # print(UX)
    UX[UX < 0] = 0
    UX[UX >= width] = width - 1
    UY[UY < 0] = 0
    UY[UY >= height] = height - 1

    mask_img = np.zeros(image.shape[:2], np.uint8)
    cv2.circle(mask_img, (eye_landmark.x, eye_landmark.y), math.ceil(radius), (255, 255, 255), -1)

    # mask
    np.copyto(UX, mapx, where=mask_img == 0)
    np.copyto(UY, mapy, where=mask_img == 0)
    # print(whereUX)
    print((UX-mapx)[244])
    UX = UX.astype(np.float32)

    UY = UY.astype(np.float32)
    # print(UX[243])
    processed_image = cv2.remap(image, UX, UY, interpolation=cv2.INTER_LINEAR)

    return processed_image


cv2.namedWindow("beautify")
cv2.resizeWindow("beautify", 640, 400)
cv2.createTrackbar("Eye amplification", "beautify", 0, 60, empty)

while True:
    if webcam:
        cap = cv2.VideoCapture(0)
        success, img = cap.read()
    else:

        # faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
        img = cv2.imread('1.jpeg')
        img_ori = img.copy()
    # img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

        landmarks = predictor(imgGray, face)
        face_points = []

        left_eye_landmark = landmarks.part(37)

        right_eye_landmark = landmarks.part(44)

        eye_landmark_down = landmarks.part(27)

        s = cv2.getTrackbarPos("Eye amplification", "beautify")
        r_r = math.sqrt(math.pow((landmarks.part(42).x - landmarks.part(45).x), 2) + math.pow(
            (landmarks.part(42).y - landmarks.part(45).y), 2))
        l_r = math.sqrt(math.pow((landmarks.part(36).x - landmarks.part(39).x), 2) + math.pow(
            (landmarks.part(36).y - landmarks.part(39).y), 2))
        # cv2.line(img,(landmarks.part(42).x,landmarks.part(42).y), (landmarks.part(45).x,landmarks.part(45).y),(255,0,0),3)
        img = bigger_eye(img, left_eye_landmark, eye_landmark_down, int(l_r),s)
        img = bigger_eye(img, right_eye_landmark, eye_landmark_down, int(r_r), s)

    cv2.imshow("s", img)
    cv2.imshow("o", img_ori)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
