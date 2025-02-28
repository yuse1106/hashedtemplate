# 入力画像に異なる背景画像にする
import cv2
import numpy as np
import os

def add(object_img, back_img):

    # HSVに変換する
    # hsv = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.int8)
    # 2値化する
    #bin_img = cv2.inRange(hsv, (0,10,0),(255,255,255))
    lower = 20
    uppper = 40
    bin_img = cv2.inRange(gray, lower, uppper)
    # cv2.imshow("bin", bin_img)
    # cv2.waitKey(0)

    # 2値化処理
    # gray = cv2.imread("train1/apple/apple_0.jpg", cv2.IMREAD_GRAYSCALE)
    # ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    # 輪郭抽出
    # contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # max_cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # マスク画像の作成
    # mask = np.zeros_like(bin_img)
    # mask_img = cv2.drawContours(mask, [max_cnt], -1, 255, thickness = cv2.FILLED)
    # cv2.imshow("mask", np.array(mask_img))
    # cv2.waitKey(0)

    mask_cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    mask = cv2.bitwise_not(bin_img)

    # 貼り付け位置
    x, y = 0, 0

    w = min(object_img.shape[1], back_img.shape[1] - x)
    h = min(object_img.shape[0], back_img.shape[0] - y)

    # 合成する領域
    object_roi = object_img[:h, :w]
    back_roi = back_img[y:y+h, x:x+w]

    # 合成する
    back_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, back_roi, object_roi.astype(float))
    # cv2.imshow("add_img", back_roi)
    # cv2.waitKey(0)
    return back_roi

if __name__ == '__main__':

    output_dir = '../test/new_input_back30/'
    os.makedirs(output_dir, exist_ok = True)

    back_img = cv2.imread('../image/back_30.jpg')
    back_img = cv2.resize(back_img, (416, 416))
    cv2.imwrite('../image/back_img_30.jpg', back_img)

    kernel = np.ones((5, 5), np.uint8)

    number = 20
    for i in range(number):
        # 画像の読み込み
        image = cv2.imread(f'../test/new_input2/input_{i}.png')
        back_img = cv2.imread('../image/back_30.jpg')
        back_img = cv2.resize(back_img, (416, 416))

        # # 白色の部分をマスクして
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # thre, mask = cv2.threshold(gray_img, 0, 50, cv2.THRESH_BINARY)
        # mask = cv2.inRange(gray_img, 254, 255)

        # # モルフォロジー
        # mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # # マスク画像の反転
        # mask_inv = cv2.bitwise_not(mask)

        # # 画像と背景を合成
        # result = cv2.bitwise_and(image, image, mask=mask_inv)
        # back_portion = cv2.bitwise_and(back_img, back_img, mask=mask_cleaned)
        # final_img = cv2.add(result, back_portion)

        final_img = add(image, back_img)

        cv2.imshow('output', final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(f'test/input_back2/input_{i}.jpg', final_img)
        cv2.imwrite(output_dir+f'input_{i}.png', final_img)
        print(final_img.shape)
