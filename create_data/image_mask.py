import cv2
import numpy as np
import glob
import os
import random
from matplotlib import pyplot as plt

def add(object_img, back_img):
    # hsvに変換する
    # hsv = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

    # plt.hist(gray.ravel(), 256, [0,256]);
    # plt.savefig("Brightness_hist.png")

    # 2値化する
    # bin_img = cv2.inRange(hsv, (10,10,10), (255,255,255))
    # 背景のマスクを作成するため，lowerとupperを変更する必要がある
    lower = 60
    upper = 255
    bin_img = cv2.inRange(gray, lower, upper)
    cv2.imshow("bin", bin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2値化処理
    # gray = cv2.imread("train1/apple/apple_0.jpg", cv2.IMREAD_GRAYSCALE)
    # ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # マスク画像の作成
    mask = np.zeros_like(bin_img)
    mask_img = cv2.drawContours(mask, [max_cnt], -1, 255, thickness=cv2.FILLED)
    cv2.imshow("mask", np.array(mask_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    center_x = int((back_img.shape[1] - object_img.shape[1]) // 2)
    center_y = int((back_img.shape[0] - object_img.shape[0]) // 2)

    # 貼り付け位置
    x, y = center_x, center_y
    # w = min(object_img.shape[1], back_img.shape[1] - x)
    # h = min(object_img.shape[0], back_img.shape[0] - y)

    # # 合成する領域
    # object_roi = object_img[:h, :w]
    # back_roi = back_img[y:y+h, x:x+w]

    # mask_resized = mask[:h,:w]

    # # 合成
    # back_roi[:] = np.where(mask_resized[:, :, np.newaxis] == 0, back_roi, object_roi)

        # 合成する領域をback_imgと同じサイズに保つ
    back_img_copy = back_img.copy()
    back_img_copy[y:y+object_img.shape[0], x:x+object_img.shape[1]] = np.where(mask[:, :, np.newaxis] == 0,
                                                                             back_img_copy[y:y+object_img.shape[0], x:x+object_img.shape[1]],
                                                                             object_img)
    return back_img_copy
    # return back_roi

def add_1(object, back):
    image = object
    back_img = back

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thre, mask = cv2.threshold(gray_img, 170, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(image, image, mask = mask_inv)
    back_portion = cv2.bitwise_and(back_img, back_img, mask = mask)
    final_img = cv2.add(result, back_portion)
    return final_img

if __name__ == '__main__':
    for i in range(1,4):
        path1 = f'../image/new_resize{i}.png'
        path2 = '../image/im1.jpg'

        object_img = cv2.imread(path1)
        back_img = cv2.imread(path2)

        # 背景画像のBGRを取得
        (b, g, r) = back_img[0,0]

        #　合成画像
        add_img = add(object_img, back_img)

        # 設定するRGB
        rgb = [r, g, b]
        true = np.all(add_img == [0,0,0], axis=-1)
        add_img[true] = rgb


        cv2.imwrite(f'../image/new_back{i}.png', add_img)
        cv2.imshow("new_back", add_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()