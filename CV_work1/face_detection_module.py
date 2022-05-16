import cv2
import mediapipe as mp
import time


class FaceDetection():
    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_confidence)

    def find_faces(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.face_detection.process(img_RGB)
        bboxs = []
        if self.res.detections:
            for id, detection in enumerate(self.res.detections):
                # mp_draw.draw_detection(img, detection)
                # mp_draw.draw_landmarks(img)
                # print(id)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxT = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxT.xmin * iw), int(bboxT.ymin * ih), \
                       int(bboxT.width * iw), int(bboxT.height * ih)

                bboxs.append([id, bbox, detection.score])
                if (draw == True):
                    img = self.decorate_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 50),
                                cv2.FONT_HERSHEY_PLAIN,
                                6, (255, 255, 0), 5)
        return img, bboxs

    def decorate_draw(self, img, bbox, l=140, t=15,rt = 3):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        cv2.line(img, (x,y), (x+l,y), (255,0,255), t)
        cv2.line(img, (x,y), (x,y+l), (255,0,255), t)
        cv2.line(img, (x1,y), (x1-l,y), (255,0,255), t)
        cv2.line(img, (x1,y), (x1,y+l), (255,0,255), t)
        cv2.line(img, (x,y1), (x+l,y1), (255,0,255), t)
        cv2.line(img, (x,y1), (x,y1-l), (255,0,255), t)
        cv2.line(img, (x1,y1), (x1-l,y1), (255,0,255), t)
        cv2.line(img, (x1,y1), (x1,y1-l), (255,0,255), t)
        return img
def main():
    cap = cv2.VideoCapture("video_repo/woman.mp4")
    previous_time = 0
    detector = FaceDetection()
    while True:
        success, img = cap.read()
        # print(img.shape)
        # img = cv2.resize(img,(0,0), None, 0.5, 0.5)
        print(img.shape)

        img, bboxs = detector.find_faces(img)
        print(bboxs)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 150), cv2.FONT_HERSHEY_PLAIN, 14, (0, 255, 0), 5)

        img = cv2.resize(img, (0, 0), None, 0.2, 0.2)
        cv2.imshow("2women", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
