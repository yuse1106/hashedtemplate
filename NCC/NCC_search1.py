import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import cv2
import numpy as np
import glob
import time
from utils import *

def ncc_template_matching(input_image, template_list, position, threshold, scale=4, search_pixel=2):
    # 粗探索画像を準備
    small_input = cv2.resize(input_image, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
    
    #detection_results = []
    
    for i, template in enumerate(template_list):
        h_t, w_t = template.shape[:2]
        # テンプレートの縮小
        small_template = cv2.resize(template, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_LINEAR)
        t_h, t_w = small_template.shape[:2]
        # NCC粗探索
        result = cv2.matchTemplate(small_input, small_template, cv2.TM_CCORR_NORMED)
        locations = np.where(result >= threshold)

        # 元の解像度に戻す
        for pt in zip(*locations[::-1]):
            x = int(pt[0] * scale)
            y = int(pt[1] * scale)

            # 精密探索
            max_score = -1
            final_x, final_y = x, y
            for dx in range(-search_pixel // 2, search_pixel // 2+1):
                for dy in range(-search_pixel // 2, search_pixel // 2+1):
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx + t_w > input_image.shape[1] or ny + t_h > input_image.shape[0]:
                        continue

                    cropped = input_image[ny:ny + h_t, nx:nx + w_t]
                    score = cv2.matchTemplate(cropped, template, cv2.TM_CCORR_NORMED)[0][0]

                    if score> max_score:
                        max_score = score
                        final_x, final_y = nx, ny
            thresholds = [0.96, 0.98, 0.97]
            if max_score > thresholds[label]:
                #detection_results.append((label, final_x, final_y, w_t, h_t, max_score))
                position[max_score] = np.array([final_x, final_y, final_x+w_t, final_y+h_t])

    return position

# バウンディングボックス
def draw_boxes(img, boxes, i, location):
    dst = img.copy()
    for x1, y1, x2, y2 in boxes:
        location.append((i, x1, y1))
        if i == 0:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        elif i == 1:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,255,255), thickness=3)
        else:
            cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,200,255), thickness=3)
    #print("number of boxes", len(boxes))
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


def iou(box1, box2):
    _, x1, y1, w1, h1, _ = box1
    _, x2, y2, w2, h2, _ = box2

    # 各ボックスの座標
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2

    # 重なり領域の座標
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    # 重なり領域の面積
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 各ボックスの面積
    area1 = (xa2 - xa1) * (ya2 - ya1)
    area2 = (xb2 - xb1) * (yb2 - yb1)

    # IoUの計算
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

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
    #overlap = (i_w * i_h) / area[indices[1:]]
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
        loc_total = correct_label.size
    else:
        total = prediction_label.size
        loc_total = prediction_label.size
    # loc_total = correct_label.size
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
        iou, index = calculation_iou(correct_location, pre, 84, 84)
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


# 使用例
if __name__ == "__main__":
     # 保存ファイル
    output_path = 'result/NCC_somitu'
    os.makedirs(output_path, exist_ok=True)

    total_time = 0
    total_time1 = 0
    total_time2 = 0
    total_time3 = 0
    total_loc_acc = 0
    total_acc = 0
    total_distance = 0
    number = 20
    for j in range(number):
        input_img = cv2.imread(f"../test/new_input2/input_{j}.png")
        path = "../new_dataset1"
        path_list = glob.glob(path+'/*')
        result_img = input_img.copy()
        all_detections = [{} for _ in range(4)]
        start_time = time.time()
        for label, pic_path in enumerate(path_list):
            template_paths = sorted(glob.glob(pic_path+"/*.png"))
            templates = [cv2.imread(path) for path in template_paths]
            if label == 0:
                threshold = 0.92
            elif label == 1:
                threshold = 0.97
            elif label == 2:
                threshold = 0.92
            position = ncc_template_matching(input_img, templates, all_detections[label],threshold)

            #all_detections.extend(detections)
        iou_threshold = 0.5
        location = []
        for k, pos in enumerate(all_detections):
            scores = np.array(list(pos.keys()))
            boxes = np.array(list(pos.values()))
            box = nms(boxes, scores, iou_threshold)
            result_img, location = draw_boxes(result_img, box, k, location)
        #filtered_detections = nms(all_detections, iou_threshold)
                # 結果描画
            #result_img = input_img.copy()
        # for i, x, y, w, h, score in filtered_detections:
        #     if label == 0:
        #         color = (0,0,255)
        #         cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        #         cv2.putText(result_img, f"Label {label}: {score:.4f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #     elif label == 1:
        #         color = (50,255,255)
        #         cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        #         cv2.putText(result_img, f"Label {label}: {score:.4f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #     elif label == 2:
        #         color = (50, 200, 255)
        #         cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        #         cv2.putText(result_img, f"Label {label}: {score:.4f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # #print(f"Elapsed time: {elapsed_time:.2f} seconds")
        # print("Detections:", filtered_detections)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        cv2.imwrite(f"result/NCC_somitu/result_{j}.jpg", result_img)

        location = np.array(location)
        sort_index = np.lexsort((location[:,1], location[:,2]))
        prediction = location[sort_index]
        # 正解
        loca = np.load('../newdata/newdata_npy/new_location2.npy')
        correct = loca[j][:,:3]

        accu_thre = 0.5
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
        total_distance += total_dis
        # print(location)
        # print(correct)
        # print(prediction)
        print(dis)
        # print("位置の正解率:", accuracy_location)
        # print("正解率：", accuracy_label)
        total_loc_acc += accuracy_location
        total_acc += accuracy_label
        total_time += elapsed_time
            # if label == 0:
            #     cv2.imwrite(f"result_{label}.jpg", result_img)
            # elif label == 1:
            #     cv2.imwrite(f"result_{label}.jpg", result_img)
                
    average_time = round(total_time / number, 3)
    #average_time1 = round(total_time1 / number, 3)
    #average_time2 = round(total_time2 / number, 3)
    #average_time3 = round(total_time3 / number, 3)
    average_pixel = round(total_distance / number, 3)
    average_loc_acc = round(total_loc_acc / number, 3)
    average_acc = round(total_acc / number, 3)
    print('平均時間：', average_time)
    print('平均ピクセル誤差：', average_pixel)
    print('平均位置精度：', average_loc_acc)
    print('平均精度：', average_acc)
