import numpy as np
import cv2
import cvzone
import mediapipe as mp
from cvzone.SelfiSegmentationModule import SelfiSegmentation

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
seg = SelfiSegmentation()


def get_center_body_coordinates(res, width, height):
    # find the center coordinate of the body
    body_coordi_list = [
        (int(res.pose_landmarks.landmark[11].x * width), int(res.pose_landmarks.landmark[11].y * height)),
        (int(res.pose_landmarks.landmark[12].x * width), int(res.pose_landmarks.landmark[12].y * height)),
        (int(res.pose_landmarks.landmark[23].x * width), int(res.pose_landmarks.landmark[23].y * height)),
        (int(res.pose_landmarks.landmark[24].x * width), int(res.pose_landmarks.landmark[24].y * height))]

    res_x, res_y = 0, 0
    for t in body_coordi_list:
        res_x, res_y = res_x + t[0], res_y + t[1]

    res_x = int(res_x / len(body_coordi_list))
    res_y = int(res_y / len(body_coordi_list))
    return res_x, res_y


frame = cv2.imread('img_repo/models/model2.jpg')
frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
frame_copy = frame.copy()
model_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
res = pose.process(model_img_rgb)

if res.pose_landmarks:
    mp_draw.draw_landmarks(frame_copy, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# define range of green color in HSV
lower_green = np.array([31, 13, 0])
upper_green = np.array([56, 54, 255])
# Threshold the HSV image to get only blue colors
mask_white = cv2.inRange(hsv, lower_green, upper_green)
mask_black = cv2.bitwise_not(mask_white)

# converting mask_black to 3 channels
W, L = mask_black.shape
mask_black_3CH = np.empty((W, L, 3), dtype=np.uint8)
mask_black_3CH[:, :, 0] = mask_black
mask_black_3CH[:, :, 1] = mask_black
mask_black_3CH[:, :, 2] = mask_black

dst3 = cv2.bitwise_and(mask_black_3CH, frame)

# ///////
W, L = mask_white.shape
mask_white_3CH = np.empty((W, L, 3), dtype=np.uint8)
mask_white_3CH[:, :, 0] = mask_white
mask_white_3CH[:, :, 1] = mask_white
mask_white_3CH[:, :, 2] = mask_white

dst3_wh = cv2.bitwise_or(mask_white_3CH, dst3)

# /////////////////

# changing for design
design = cv2.imread('img_repo/clothes/tshirto.jpg')
design = cv2.resize(design, mask_black.shape[1::-1])
affine_arr = np.float32([[1, 0, 0], [0, 1, -25]])
design = cv2.warpAffine(design, affine_arr, (design.shape[1], design.shape[0]))
design[-30:] = 255, 255, 255
design_mask_mixed = cv2.bitwise_or(mask_black_3CH, design)

final_mask_black_3CH = cv2.bitwise_and(design_mask_mixed, dst3_wh)

# get center coordinate
res_x, res_y = get_center_body_coordinates(res, frame.shape[1],frame.shape[0])
tshirt_color = design[res_y, res_x]
mask_color_3CH = np.empty((W, L, 3), dtype=np.uint8)
mask_color_3CH[:,:] = tshirt_color
gray_img = cv2.cvtColor(final_mask_black_3CH, cv2.COLOR_BGR2GRAY)

combo = [frame, mask_black_3CH, dst3, mask_white_3CH, dst3_wh, design, design_mask_mixed,
         final_mask_black_3CH,mask_color_3CH]

for i in (254, 240):
    ret, mask_whitehole = cv2.threshold(final_mask_black_3CH,i, 255, cv2.THRESH_BINARY)
    mask_blackhole = ~mask_whitehole
    mask_b = np.zeros_like(mask_color_3CH)

    mask_sup = cv2.bitwise_or(mask_color_3CH, mask_blackhole)
    mask_end = cv2.bitwise_and(final_mask_black_3CH, mask_sup)
    combo.extend([mask_whitehole, mask_blackhole, mask_sup, mask_end])
    if i == 240:
        bg_img = cv2.imread("img_repo/backgrounds/01_background.jpg")
        bg_img = cv2.resize(bg_img, mask_black.shape[1::-1])
        mask_end_bk = seg.removeBG(mask_end,bg_img, threshold=0.5)
        combo.append(mask_end_bk)

# final_mask_black_3CH =
combo_img = cvzone.stackImages(combo, 6, 1)
cv2.imshow("combo", combo_img)
cv2.waitKey()
