import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video_repo/several_people.mp4")
previous_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()
while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = face_detection.process(img_RGB)

    if res.detections:
        for id, detection in enumerate(res.detections):
            # mp_draw.draw_detection(img, detection)
            # mp_draw.draw_landmarks(img)
            # print(id)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxT = detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxT.xmin * iw), int(bboxT.ymin * ih), \
                    int(bboxT.width * iw), int(bboxT.height * ih)

            cv2.rectangle(img, bbox, (255,0,255), 4)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-50), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 0), 5)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img,f'FPS: {int(fps)}', (20,150), cv2.FONT_HERSHEY_PLAIN, 14, (0,255,0), 5)



    img = cv2.resize(img, (0, 0), None, 0.2, 0.2)
    cv2.imshow("2women", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
