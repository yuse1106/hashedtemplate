# import cv2 
# import numpy as np
# from Ipython.display import Image, display

# def imshow(img):
#     ret, encoded = cv2.imencode(".jpg", img)
#     display(Image(encoded))

# # 物体の画像を読み込む
# object_img = cv2.imread("train1/apple/apple_0.jpg")

# # 背景画像を読み込む
# back_img = cv2.imread("train2/w_back/w_back_0.jpg")

# # HSVに変換する
# hsv = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)

# # 2値化する
# bin_img = cv2.inRange(hsv, (0,10,0),(255,255,255))
# cv2.imshow("bin", bin_img)
# cv2.waitKey(0)

# # 輪郭抽出する
# contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 面積が最大の輪郭を取得する
# contour = max(contours, key=lambda x: cv2.contourArea(x))

# # マスク画像を生成する
# mask = np.zeros_like(bin_img)
# cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

# # 貼り付け位置
# x, y = 10, 10

# w = min(object_img.shape[1], back_img.shape[1] - x)
# h = min(object_img.shape[0], back_img.shape[0] - y)

# # 合成する領域
# object_roi = object_img[:h, :w]
# back_roi = back_img[y:y+h, x:x+w]

# # 合成する
# back_roi[:] = np.where(mask[:h, :w, np.newaxis] == 0, back_roi, object_roi)
# imshow(back_roi)

import cv2
import numpy as np
import glob
import os 
import random

def add(object_img, back_img):

    # HSVに変換する
    # hsv = cv2.cvtColor(object_img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5,5), np.int8)
    # 2値化する
    #bin_img = cv2.inRange(hsv, (0,10,0),(255,255,255))
    lower = 200
    uppper = 255
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



    # # 画像合成前処理
    # object_img[mask<255]=[0,0,0]
    # back_img[mask==255] = [0,0,0]
    # cv2.imshow("img", np.array(object_img))

    # # 画像合成
    # add_img = cv2.add(back_img, object_img)
    # cv2.imshow("add_img", np.array(add_img))
    # cv2.waitKey(0)

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

def add_1(object, back):
    image = object
    back_img = back

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thre, mask = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(image, image, mask = mask_inv)
    back_portion = cv2.bitwise_and(back_img, back_img, mask = mask)
    final_img = cv2.add(result, back_portion)
    return final_img

if __name__ == '__main__':
    path = '../add_dataset/add_train3'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path+'/apple/', exist_ok=True)
    os.makedirs(path+'/lemon/', exist_ok=True)
    os.makedirs(path+'/orange/', exist_ok=True)

    path1 = '../tra1'
    path2 = '../add_dataset/w_back'
    path_list = glob.glob(path1+'/*')
    for label, pic_path in enumerate(path_list):
        list = glob.glob(pic_path+'/*')
        back_list = glob.glob(path2+'/*')
        # 合成する画像の枚数
        number = 90
        for i in range(number):
            # ランダムに抽出する
            data = random.choice(list)
            data1 = random.choice(back_list)
            # すべてのファイルパスにする
            #data1 = os.path.split(data)[1]
            # 画像の読み込み
            #object_img = cv2.imread("train1/apple/apple_0.jpg")
            object_img = cv2.imread(data)
            #back_img = cv2.imread("train2/w_back/w_back_0.jpg")
            back_img = cv2.imread(data1)

            # 合成画像
            add_img = add(object_img, back_img)
            # cv2.imshow("add_img", add_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 画像サイズ変更
            # add_img = cv2.resize(add_img, (84,84))

            if label == 0:
                cv2.imwrite(path+f'/apple/back1_{i}.jpg', add_img)
            elif label == 1:
                cv2.imwrite(path+f'/lemon/back2_{i}.jpg', add_img)
            elif label == 2:
                cv2.imwrite(path+f'/orange/back3_{i}.jpg', add_img)
    print("finish")
