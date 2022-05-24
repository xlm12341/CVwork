import mediapipe as mp
import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import cvzone


def get_center_body_coordinates(res):
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


def get_contours(binary_img, img, circumference):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_list = []
    is_first = True
    mask_img = np.zeros_like(img)

    for con in contours:
        area = cv2.contourArea(con)
        print(area)
        if is_first:
            is_first = False
            continue
        if area > circumference:
            mask = cv2.fillPoly(mask_img, [con], (255, 255, 255))

            cv2.drawContours(img, [con], -1, (0, 0, 255), 1)
    return mask_img, img


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
seg = SelfiSegmentation()

# prep images
background_list = os.listdir("img_repo/backgrounds")
clothes_list = os.listdir("img_repo/clothes")
models_list = os.listdir("img_repo/models")

b_img_list = []
for d in background_list:
    img = cv2.imread(f"img_repo/backgrounds/{d}")
    b_img_list.append(img)
c_img_list = []
for d in clothes_list:
    img = cv2.imread(f"img_repo/clothes/{d}")
    c_img_list.append(img)
m_img_list = []
for d in models_list:
    img = cv2.imread(f"img_repo/models/{d}")
    m_img_list.append(img)

# process all the image sets
for i in range(len(background_list)):
    model_img = m_img_list[i]
    model_img = cv2.resize(model_img, (0, 0), None, 0.5, 0.5)
    ori_img = model_img.copy()

    height, width, channel = model_img.shape
    model_img_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
    white = (255, 255, 255)
    res = pose.process(model_img_rgb)

    if res.pose_landmarks:
        mp_draw.draw_landmarks(model_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # blur 
    blurred_img = cv2.blur(ori_img, (5, 5))

    # prepare the mask, 2pixel larger in height an width
    mask = np.zeros((height + 2, width + 2), np.uint8)

    # flood fill from corners
    cv2.floodFill(blurred_img, mask, (10, 10), white, (10, 10, 10), (10, 10, 10), cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(blurred_img, mask, (int(width * 9 / 10), 10), white, (10, 10, 10), (10, 10, 10),
                  cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(blurred_img, mask, (10, int(height * 9 / 10)), white, (10, 10, 10), (10, 10, 10),
                  cv2.FLOODFILL_FIXED_RANGE)

    # cv2.circle(blured_img, (res_x, res_y), 5,
    #            (0, 255, 0), 3)
    # cv2.floodFill(blured_img, mask, (res_x, res_y), (255, 0, 0), (30, 30, 30), (30, 30, 30), cv2.FLOODFILL_FIXED_RANGE)

    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    # the head in some given images is cropped
    gray_img[0:2, :] = 255

    # cut lower body 
    gray_img[int(res.pose_landmarks.landmark[23].x * width) - width:-1, :] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 开闭运算，先开运算去除背景噪声，再继续闭运算填充目标内的孔洞
    opened = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # important
    ret, binary_img = cv2.threshold(closed, 200, 255, cv2.THRESH_BINARY)

    # remove the head

    tmp_point = ((int(res.pose_landmarks.landmark[11].x * width) + int(res.pose_landmarks.landmark[12].x * width)) // 2 \
                     , (int(res.pose_landmarks.landmark[11].y * height) + int(
        res.pose_landmarks.landmark[12].y * height)) // 2)

    nose_point = res.pose_landmarks.landmark[0].x, res.pose_landmarks.landmark[0].y
    # cut!
    # new point between tmp and nose point
    cut_point = (int(tmp_point[0] * 0.7 + nose_point[0] * 0.3), int(tmp_point[1] * 0.8 + nose_point[1] * 0.2))
    binary_img[:cut_point[1], :] = 255

    # get contours

    # cv2.drawContours(img, contours, -1, (0, 0, 255), -1)

    mask_img, model_img = get_contours(binary_img, model_img, 10000)

    cloth_img = c_img_list[i]
    cloth_img = cv2.resize(cloth_img, (0, 0), None, 0.5, 0.5)
    cloth_img_ori = cloth_img.copy()

    cloth_gray_img = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2GRAY)
    ret, cloth_binary_img = cv2.threshold(cloth_gray_img, 200, 255, cv2.THRESH_BINARY)

    cloth_mask, cloth_img = get_contours(cloth_binary_img, cloth_img, 3000)

    cloth_cut_img = cv2.bitwise_and(cloth_mask, cloth_img_ori)

    # resize to model_img
    cloth_cut_img = cv2.resize(cloth_img, (width, height))
    cloth_cut_resize_img = cv2.bitwise_and(cloth_cut_img, mask_img)

    ready_img = cv2.bitwise_and(ori_img, ~mask_img)
    finish_img = cv2.bitwise_or(ready_img, cloth_cut_resize_img)

    # remove background
    background_img = b_img_list[i]
    background_img = cv2.resize(background_img, (width, height))
    background_finish = seg.removeBG(finish_img, background_img, threshold=0.4)

    # total_img = [blurred_img, ori_img, closed, binary_img, mask_img, model_img, cloth_binary_img, cloth_img, cloth_mask,
    #              cloth_cut_img, cloth_cut_resize_img, ready_img, finish_img,background_finish]
    #
    # total_img = cvzone.stackImages(total_img, len(total_img) // 2, 1)

    cv2.imshow(f"{i}",background_finish)
cv2.waitKey(0)
