import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
import math
import glob
import time
import os
from scipy.spatial import distance
from utils import *


# バウンディングボックス
def draw_boxes(img, boxes, i, location):
    dst = img.copy()
    for x1, y1, x2, y2 in boxes:
        location.append((i, x1, y1))
        dst = rectangle(dst, i, x1, y1, x2, y2)
    return dst, location

# nms
def nms(boxes, scores, overlap_thresh):
    if len(boxes) <= 1:
        return boxes

    # float 型に変換する。
    #boxes = boxes.astype("float")

    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(-scores)  # スコアを降順にソートしたインデックス一覧
    selected = []  # NMS により選択されたインデックス一覧

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # indices は降順にソートされているので、一番最後の要素の値 (インデックス) が
        # 残っている中で最もスコアが高い。
        #last = len(indices) - 1

        #current_index = indices[last]
        #indices[1:] = indices[:last]
        current_index = indices[0]
        selected.append(current_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[current_index], x1[indices[1:]])
        i_y1 = np.maximum(y1[current_index], y1[indices[1:]])
        i_x2 = np.minimum(x2[current_index], x2[indices[1:]])
        i_y2 = np.minimum(y2[current_index], y2[indices[1:]])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        #overlap = (i_w * i_h) / area[indices[1:]]
        intersection = i_w * i_h
        union =  area[current_index] + area[indices[1:]] - intersection
        overlap = intersection / union


        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        #indices = np.delete(
        #    indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        #)
        indices = indices[np.where(overlap <= overlap_thresh)[0] + 1]

    # 選択された短形の一覧を返す。
    return boxes[selected].astype("int")


def calculation_iou(true_location, prediction, w_e, h_e):
    dis = 1000
    index = 0
    x2, y2 = prediction
    for i in range(len(true_location)):
        x_1, y_1 = true_location[i]
        distance = abs(x_1-x2) + abs(y_1-y2)
        if distance <=dis:
            dis = distance
            index = i
    x1, y1 = true_location[index]
    w, h = w_e, h_e
    # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
    i_x1 = max(x1, x2)
    i_y1 = max(y1, y2)
    i_x2 = min(x1+w, x2+w)
    i_y2 = min(y1+h, y2+h)

    # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
    # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
    i_w = max(0, i_x2 - i_x1 + 1)
    i_h = max(0, i_y2 - i_y1 + 1)

    # 選択した短形と残りの短形の Overlap Ratio を計算する。
    #overlap = (i_w * i_h) / area[remaining_indices]
    overlap = (i_w * i_h) / (2 * (w * h) - i_w * i_h)
    return overlap, index

def pixel_dis(true_location, prediction):
    x1, y1 = true_location
    x2, y2 = prediction
    distance = abs(x1-x2) + abs(y1-y2)
    return distance

def accuracy(correct, prediction, thre_accu, distance_thre):
    correct_label = correct[:,0]
    correct_location = correct[:, 1:]
    prediction_label = prediction[:,0]
    prediction_location = prediction[:,1:]
    location_count = 0
    label_count = 0
    if prediction_label.size < correct_label.size:
        total = correct_label.size
    else:
        total = prediction_label.size
    loc_total = correct_label.size
    dis = []
    total_dis = 0
    # for i, (true, pre) in enumerate(zip(correct_location, prediction_location)):
    #     iou = cal_iou(true, pre, w_e, h_e)
    #     if iou >= thre_accu:
    #         distance = pixel_dis(true, pre)
    #         dis.append(distance)
    #         if distance <= distance_thre:
    #             location_count += 1
    #         if correct_label[i] == prediction_label[i]:
    #             label_count += 1
    for i, pre in enumerate(prediction_location):
        # true = correct_location[i,:]
        # if correct_label[i] == prediction_label[i]:
        #     label_count += 1
        #     distance = pixel_dis(true, pre)
        #     total_dis += distance
        #     dis.append(distance)
        #     if distance <= distance_thre:
        #         location_count += 1
        iou, index = calculation_iou(correct_location, pre, w, h)
        true = correct_location[index]
        if iou >= thre_accu:
            # distance = pixel_dis(true, pre)
            # total_dis += distance
            # dis.append(distance)
            # if distance <= distance_thre:
            #     location_count += 1
            if correct_label[index] == prediction_label[i]:
                label_count += 1
                distance = pixel_dis(true, pre)
                total_dis += distance
                dis.append(distance)
                if distance <= distance_thre:
                    location_count += 1
    total_dis = total_dis / len(dis)
    # 位置の正確率
    accuracy_location = (location_count / loc_total) * 100
    # 分類の正解率
    accuracy_label = (label_count / total) * 100
    return accuracy_location, accuracy_label, dis, total_dis


if __name__ == '__main__':

     # 保存ファイル
    output_path = 'result/NCC_output'
    out_path = 'result/NCC_out'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    total_time = 0
    total_time1 = 0
    total_time2 = 0
    total_time3 = 0
    total_loc_acc = 0
    total_acc = 0
    total_distance = 0
    number = 20
    for j in range(number):
        start_time = time.time()
        input_image = cv2.imread(f"../test/new_input2/input_{j}.png")

        #template
        path = "../new_dataset1"
        path_list = glob.glob(path+'/*')
        #path_list = ["apple", "lemon", "orange"]
        
        train_x = []
        train_y = []
        apple_posi ={}
        lemon_posi = {}
        orange_posi ={}
        posi = []
        detected_objects_image = input_image.copy()
        value_max = 0
        for label, pic_path in enumerate(path_list):
            #pic_path = glob.glob(path + "/" + pic_path)
            train_list = glob.glob(pic_path+'/*.png')
            train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
            for label_i, img_path in enumerate(train_list):
                template_image = cv2.imread(img_path)
                # cv2.imshow("template",template_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                if template_image is None:
                    print("Failed to read template Image:")
                    continue
                # NCC

                result = cv2.matchTemplate(detected_objects_image, template_image, cv2.TM_CCORR_NORMED)
                ma = result.max()
                if ma >= value_max:
                    value_max = ma
                # res = []

                # threshold
                threshold = 0.99
                location = np.where(result >= threshold)

                for pt in zip(*location[::-1]):
                    x = pt[1]
                    y = pt[0]
                    w, h, _= template_image.shape
                    # flat_index = np.argmax(result)
                    # max_index = np.unravel_index(flat_index, result.shape)
                    # print(max_index)
                    # print(result[max_index])
                    if label == 0:
                        # print(label_i)
                        value = result[x,y]
                        apple_posi[value] = np.array([y, x, y+ h, x + w])
                        input_image = cv2.rectangle(input_image, (y, x), (y+h, x+w), (0, 0, 255), 4)
                    elif label == 1:
                        value = result[x,y]
                        lemon_posi[value] = np.array([y, x, y+ h, x + w])
                        input_image = cv2.rectangle(input_image, (y, x), (y+h, x+w), (50, 255, 255), 4)
                    elif label == 2:
                        value = result[x,y]
                        orange_posi[value] = np.array([y, x, y+ h, x + w])
                        input_image = cv2.rectangle(input_image, (y, x), (y+h, x+w), (50, 200, 255), 4)
                    #cv2.rectangle(detected_objects_image, pt, (y+template_image.shape[1], x+template_image.shape[0]), (0,0,255), 4)
                # _, threshold_result = cv2.threshold(result, threshold, 1.0, cv2.THRESH_BINARY)

        #nms スコア量子化誤差
        posi.append(apple_posi)
        posi.append(lemon_posi)
        posi.append(orange_posi)
        location = []
        overlap_thresh = 0.5
        for k, pos in enumerate(posi):
            scores = np.array(list(pos.keys()))
            boxes = np.array(list(pos.values()))
            box = nms(boxes, scores, overlap_thresh)
            detected_objects_image, location = draw_boxes(detected_objects_image, box, k, location)

        print(value_max)
        cv2.imwrite("out.jpg", detected_objects_image)
        cv2.imwrite(f'result/NCC_output/output_{j}.jpg', input_image)
        cv2.imwrite(f'result/NCC_out/out_{j}.jpg', detected_objects_image)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)
        total_time += elapsed_time
        # 正解位置
        loca = np.load('../newdata/newdata_npy/new_location2.npy')
        correct = loca[j][:,:3]
        #correct = np.load('npy/placed_objects.npy')
        # 位置
        location = np.array(location)
        sort_index = np.lexsort((location[:,1], location[:,2]))
        prediction = location[sort_index]
        # print(correct)
        # print('##')
        # print(prediction)
        # 閾値
        accu_thre = 0.5
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
        total_distance += total_dis
        # print(location)
        # print(correct)
        # print(prediction)
        print(dis)
        print("位置の正解率:", accuracy_location)
        print("正解率：", accuracy_label)
        total_loc_acc += accuracy_location
        total_acc += accuracy_label

    average_time = total_time / number
    average_pixel = total_distance / number
    average_loc_acc = total_loc_acc / number
    average_acc = total_acc / number
    print('平均時間：', average_time)
    print('平均ピクセル誤差：', average_pixel)
    print('平均位置精度：', average_loc_acc)
    print('平均精度：', average_acc)