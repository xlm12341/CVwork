import cv2
import dlib
import face_recognition
import utils
import os
img_elon = face_recognition.load_image_file('img_repo/musk.jpg')
img_elon_test = face_recognition.load_image_file('img_repo/musk2.jpg')

img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)
img_elon_test = cv2.cvtColor(img_elon_test, cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(img_elon)[0]
encode_elon = face_recognition.face_encodings(img_elon)[0]
cv2.rectangle(img_elon, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

face_loc_test = face_recognition.face_locations(img_elon_test)[0]
encode_elon_test = face_recognition.face_encodings(img_elon_test)[0]
cv2.rectangle(img_elon_test, (face_loc_test[3], face_loc_test[0]), (face_loc_test[1], face_loc_test[2]), (255, 0, 255),
              2)
# cv2.imshow("Elon musk", img_elon)
# cv2.imshow("Elon musk2", img_elon_test)

result = face_recognition.compare_faces([encode_elon], encode_elon_test)
face_distance = face_recognition.face_distance([encode_elon], encode_elon_test)
print(result)
print(face_distance)
cv2.putText(img_elon, f'{result} {round(face_distance[0],2)}',(face_loc[3], face_loc[0]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
img_stack = utils.stackImages(1, ([img_elon], [img_elon_test]))
cv2.imshow("Elon", img_stack)
cv2.waitKey(0)
