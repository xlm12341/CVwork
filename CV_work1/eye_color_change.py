import cv2
import numpy as np
import dlib
import utils
webcam = False

cap = cv2.VideoCapture(0)


def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", 'BGR', 0, 255, empty)
cv2.createTrackbar("Green", 'BGR', 0, 255, empty)
cv2.createTrackbar("Red", 'BGR', 0, 255, empty)

def create_box(img, points, scale=5, masked=False, cropped=True):
    mask = None
    if masked is True:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        img = cv2.bitwise_and(img, mask)

        # cv2.imshow("eye",img)
    if cropped is True:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        img_crop = img[y:y + h, x:x + w]
        img_crop = cv2.resize(img_crop, (0, 0), None, scale, scale)
        return img_crop
    return mask


while True:

    if webcam:
        success, img = cap.read()
    else:

        # faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
        img = cv2.imread('img_repo/01.jpg')
    # img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    img_original = img.copy()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    eyes_detected_mask = None


    for face in faces:
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(img_original, (x1, y1), (x2, y2), (255, 255, 0), 2)

        landmarks = predictor(imgGray, face)
        facial_points = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            facial_points.append([x, y])
            cv2.circle(img_original, (x, y), 5, (50, 50, 255), cv2.FILLED)
            cv2.putText(img_original, str(n), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

        facial_points = np.array(facial_points)
        # cropped=False , masked=True 返回 mask
        # cropped=True , masked=False 返回 rect crop
        # cropped=True , masked=True 返回 masked crop
        img_left_eye = create_box(img, facial_points[36:42], masked=True, cropped=False)
        img_right_eye = create_box(img, facial_points[42:48], masked=True,cropped=False)

        colored_eye_img = np.zeros_like(img_left_eye)
        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        # colored_eye_img[:] = 153, 0, 157
        colored_eye_img[:] = b, g, r
        print(b,g,r)

        # 双眼mask
        eyes_img = cv2.bitwise_or(img_right_eye, img_left_eye)
        if eyes_detected_mask is None:
            eyes_detected_mask = eyes_img
        else:
            eyes_detected_mask = cv2.bitwise_or(eyes_img, eyes_detected_mask)
        colored_eye_img = cv2.bitwise_and(colored_eye_img, eyes_detected_mask)

        colored_eye_img = cv2.GaussianBlur(colored_eye_img, (7, 7), 10)
        colored_eye_img = cv2.addWeighted(img, 1, colored_eye_img, 0.4, 0)
        # cv2.imshow("BGR", colored_eye_img)


        # combo_img = utils.stackImages(1, ([colored_eye_img,img_original]))
        # cv2.imshow("comb",combo_img)
        cv2.imshow("colored", colored_eye_img)
        cv2.imshow("img", img_original)

    # cv2.imshow("img org", img_original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(imgGray,1.1,4)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


# detector = dlib.get_frontal_face_detector()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = detector(imgGray)
#

#
# cv2.imshow("Result", img)
# cv2.waitKey(0)
