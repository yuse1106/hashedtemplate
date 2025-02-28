import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
import math
import time
from scipy.spatial import distance
from utils import *

# 矩形を描画，量子化誤差と位置を辞書として保存
def rectangle_candidate(image, position, error, i, cor_label):
    x1 = int(i % h1)
    y1 = int(i / h1)
    x2 = int(x1 + w_e)
    y2 = int(y1 + h_e)
    position[error] = np.array([x1, y1, x2, y2])
    image = rectangle(image, cor_label, x1, y1, x2, y2)

    return image

# バウンディングボックス
def draw(img, boxes, i, location):
    # dst = img.copy()
    color_map = [(0,0,255), (50,255,255), (50,200,255), (255,255,0), (255,0,255),(0,255,255)]
    for x1, y1, x2, y2 in boxes:
        color = color_map[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)
        location.append((i, x1, y1))

    return img, location


def draw_boxes(img, boxes, labels, location):
    color_map = [(0,0,255), (50,255,255), (50,200,255), (255,255,0), (255,0,255),(0,255,255)]
    for box, label in zip(boxes, labels):
        color = color_map[label]
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)
        location.append((label, x1, y1))
    
    return img, location
 
# 姿勢推定
# def draw_boxespose(img, boxes, i, loc):
#     dst = img.copy()
#     for x1, y1, x2, y2 in boxes:
#         loc.append((i, x1, y1))
#         location = y1 * h1 + x1
#         feature = F[location,:]
#         # 中心位置
#         x_center, y_center = (x1+x2)//2, (y1+y2)//2
#         line_length = 50
#         if i == 0:
#             pos = rotate_a(feature, apple_train)
#             posi = np.radians(pos)
#             x_end = int(x_center - line_length * np.sin(posi))
#             y_end = int(y_center - line_length * np.cos(posi))
#             cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
#             cv2.line(dst, (x_center, y_center), (x_end, y_end), color=(0,0,255), thickness=2)
#         elif i == 1:
#             pos = rotate_l(feature, lemon_train)
#             posi = np.radians(pos)
#             x_end = int(x_center - line_length * np.sin(posi))
#             y_end = int(y_center - line_length * np.cos(posi))
#             cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,255,255), thickness=2)
#             cv2.line(dst, (x_center, y_center), (x_end, y_end), color = (50, 255, 255), thickness=2)
#         else:
#             pos = rotate_o(feature, orange_train)
#             posi = np.radians(pos)
#             x_end = int(x_center - line_length * np.sin(posi))
#             y_end = int(y_center - line_length * np.cos(posi))
#             cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,200,255), thickness=2)
#             cv2.line(dst, (x_center, y_center), (x_end, y_end), color=(50,200,255), thickness=2)
#     #print("number of boxes", len(boxes))
#     return dst, loc

# IoUの計算
def cal_iou(true_location, prediction, w_e, h_e):
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

# 精度
def accuracy(correct, prediction, thre_accu, distance_thre):
    correct_label = correct[:,0]
    correct_location = correct[:, 1:]
    prediction_label = prediction[:,0]
    prediction_location = prediction[:,1:]
    location_count = 0
    label_count = 0
    # 候補点が多い時と少ない時で分母が変わる
    if prediction_label.size < correct_label.size:
        total = correct_label.size
    else:
        total = prediction_label.size
    loc_total = correct_label.size
    dis = []
    total_dis = 0
    for i, pre in enumerate(prediction_location):
        iou, index = cal_iou(correct_location, pre, w_e, h_e)
        true = correct_location[index]
        # IoUが閾値以上の場合に精度とピクセル誤差を計算
        if iou >= thre_accu:
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
    eigenFaces = np.load('new_npy/new_eigen1.npy')
    # eigenFaces = np.load('new_npy/new_eigen2_16.npy')
    X_train_seq = np.load('new_npy/X_train_seq.npy')
    y_train_seq = np.load('new_npy/y_train_seq.npy')

    # 保存ファイル
    output_path = 'result/output'
    out_path = 'result/out'
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
    label_num = len(y_train_seq[0])
    for j in range(number):
        start_time = time.time()
        #実装，入力画像
        #image = cv2.imread(f'../test/simple/color_jpg/input_{j}.jpg')
        image = cv2.imread(f'../test/simple/color_jpg_back/input_{j}.jpg')

        image_c = np.float32(image) / 255
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('image.jpg', image_back)
        img = image.copy()
        results = []
        # 特徴ベクトル生成
        for eigenFace in eigenFaces:
            res = cv2.matchTemplate(image_c, eigenFace, cv2.TM_CCORR)
            #res = cv2.matchTemplate(image, eigenFace, cv2.TM_CCOEFF)
            result = res.flatten()
            results.append(result)
        w_e, h_e = eigenFace.shape[:2]
        results = np.array(results)
        w1, h1 = res.shape
        w, h = results.shape
        end_time = time.time()

        F = np.transpose(results)
        start_time1 = time.time()
        #  # #カーネル
        # Ntrain = train_x.shape[0]
        # n_anchor = Ntrain
        # index = np.random.choice(Ntrain, n_anchor, replace=False)
        # X_anchor = train_x[index,:]
        # #ガウスカーネル
        # Vsigma = 0.3
        # features_k = np.exp(-sqdist(F, X_anchor) / (2*Vsigma*Vsigma))

        # 符号関数
        vH = np.sign(F)
        vH = vH.astype(int)
        end_time1 = time.time()

        classify = []
        # ラベルの数だけ辞書を作成
        posi_list = [{} for _ in range(label_num)]
        count = 0
        start_time2 = time.time()
        i = 0
        while i < h:    
            #1ピクセルごとにバイナリビット実行
            # バイナリビットに変更する前
            feature = F[i,:]
            # バイナリビットに変更後
            features = vH[i,:]

            n2, bits = X_train_seq.shape

            for n in range(n2):
                x = X_train_seq[n, :]
                dis = hamming_dis(features, x)
                # dis = distance.hamming(features, x)
                # dis = dis * len(x)
                if dis == 0:
                    true = 1
                    cor_label = y_train_seq[0][n]
                    break
                else:
                    true = 0
                    cor_label = 4

            # 閾値処理，num_bitのビット数によって適切な値を設定する必要がある
            threshold = 0.3
            if true == 0:
                classify.append(4)
                i += 1
            elif cor_label == 3:
                classify.append(3)
                i += 1
            else:
                classify.append(cor_label)
                error = quantization_error(feature, features)
                if error <= threshold:
                    if cor_label == 0:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 1:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 2:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                i += 1

        classify = np.array(classify)
        print(count)
        #nms スコア量子化誤差
        overlap_thresh = 0.3
        location = []
        # # ラベルごとに対してNMS
        # for k, pos in enumerate(posi_list):
        #     scores = np.array(list(pos.keys()))
        #     boxes = np.array(list(pos.values()))
        #     box, score = nms(boxes, scores, overlap_thresh)
        #     img, location = draw(img, box, k, location)
        #     #img = draw_boxespose(img, box, i)

         # nmsをラベルごとに行わずに全候補に対して
        all_scores = []
        all_boxes = []
        all_labels = []

        for k, pos in enumerate(posi_list):
            all_scores.extend(pos.keys())
            all_boxes.extend(pos.values())
            all_labels.extend([k]*len(pos))
        
        all_scores = np.array(all_scores)
        all_boxes = np.array(all_boxes)
        all_labels = np.array(all_labels)

        final_boxes, final_labels = nms_all(all_boxes, all_scores, all_labels, overlap_thresh)

        img, location = draw_boxes(img, final_boxes, final_labels, location)

        #実行時間
        end_time2 = time.time()
        elapsed_time = end_time - start_time
        elapsed_time1 = end_time1 - start_time1
        elapsed_time2 = end_time2 - start_time2
        # print("実行時間:", elapsed_time, elapsed_time1, elapsed_time2)
        total_time += elapsed_time + elapsed_time1 + elapsed_time2
        total_time1 += elapsed_time
        total_time2 += elapsed_time1
        total_time3 += elapsed_time2
        cv2.imwrite(f'result/output/output_{j}.jpg', image)
        cv2.imwrite(f'result/out/out_{j}.jpg', img)

        # 正解位置
        loca = np.load('npy/location_jpg.npy')
        correct = loca[j][:,:3]
        # print(correct)
        # 予測位置
        location = np.array(location)
        sort_index = np.lexsort((location[:,1], location[:,2]))
        prediction = location[sort_index]
        # 閾値,IoU以上で正解とみなす
        accu_thre = 0.6
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
        total_distance += total_dis
        print(dis)
        # print("位置の正答率:", accuracy_location)
        # print("正解率：", accuracy_label)
        total_loc_acc += accuracy_location
        total_acc += accuracy_label
    
    average_time = round(total_time / number, 3)
    average_time1 = round(total_time1 / number, 3)
    average_time2 = round(total_time2 / number, 3)
    average_time3 = round(total_time3 / number, 3)
    average_pixel = round(total_distance / number, 3)
    average_loc_acc = round(total_loc_acc / number, 3)
    average_acc = round(total_acc / number, 3)
    print('平均時間：', average_time, average_time1, average_time2, average_time3)
    print('平均ピクセル誤差：', average_pixel)
    print('平均位置精度：', average_loc_acc)
    print('平均精度：', average_acc)
