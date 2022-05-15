import cv2
import dlib
import face_recognition
import utils
import os
import numpy as np

path = 'img_repo'
img_list = []
classname = []
encode_list_known = []
dir_list = os.listdir(path)

for d in dir_list:
    img = cv2.imread(f'{path}/{d}')
    img_list.append(img)
    classname.append(os.path.splitext(d)[0])
print(classname)
print(dir_list)


def get_encodings(imgs):
    img_encoding_list = []
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encoding = face_recognition.face_encodings(img)[0]
        img_encoding_list.append(img_encoding)
    return img_encoding_list


encode_list_known = get_encodings(img_list)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 1, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_cur_frame = face_recognition.face_locations(img)
    faces_encodings_cur_frame = face_recognition.face_encodings(img, faces_cur_frame)

    for face_encode, face_loc in zip(faces_encodings_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, face_encode)
        face_dis = face_recognition.face_distance(encode_list_known, face_encode)
        match_index = np.argmin(face_dis)
        print(face_dis)
        print(match_index)

        if matches[match_index]:
            name = classname[match_index].upper()
            print(name)
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0),2)
            cv2.rectangle(img, (x1, y2+35),(x2, y2),(0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break