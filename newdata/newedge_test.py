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

def rectangle_candidate(image, position, error, i, cor_label):
    x1 = int(i % h1)
    y1 = int(i / h1)
    x2 = int(x1 + w_e)
    y2 = int(y1 + h_e)
    position[error] = np.array([x1, y1, x2, y2])
    image = rectangle(image, cor_label, x1, y1, x2, y2)

    return image

def matchtemp(image, eigenFaces):
    image_norm = np.float32(image) / 255.0
    results = []
    for eigenFace in eigenFaces:
        res = cv2.matchTemplate(image_norm, eigenFace, cv2.TM_CCORR)
        result = res.flatten()
        results.append(result)
    results = np.array(results)

    return results, res

# バウンディングボックス
def draw(img, boxes, i, location):
    # dst = img.copy()
    color_map = [(0,0,255), (50,255,255), (50,200,255), (255,255,0), (255,0,255),(0,255,255)]
    for x1, y1, x2, y2 in boxes:
        color = color_map[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)
        location.append((i, x1, y1))
        # if i == 0:
        #     cv2.rectangle(dst, (x1, y1), (x2, y2), color=(0,0,255), thickness=4)
        # elif i == 1:
        #     cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,255,255), thickness=4)
        # elif i == 2:
        #     cv2.rectangle(dst, (x1, y1), (x2, y2), color=(50,200,255), thickness=4)

    return img, location


def draw_boxes(img, boxes, labels, location):
    color_map = [(0,0,255), (50,255,255), (50,200,255), (255,255,0), (255,0,255),(0,255,255)]
    for box, label in zip(boxes, labels):
        color = color_map[label]
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)
        location.append((label, x1, y1))
    
    return img, location
 
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
        iou, index = cal_iou(correct_location, pre, w_e, h_e)
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
    edge_x = np.load('newdata_npy/sobel_x.npy')
    edge_y = np.load('newdata_npy/sobel_y.npy')
    # eigenFaces = np.load('newdata_npy/new_eigenFaces.npy')
    new_eigenFaces_x = np.load('newdata_npy/new_eigenFaces_x.npy')
    new_eigenFaces_y = np.load('newdata_npy/new_eigenFaces_y.npy')
    # trans = np.load('newdata_npy/trans.npy')
    X_train_seq = np.load('newdata_npy/X_train_seq.npy')
    y_train_seq = np.load('newdata_npy/y_train_seq.npy')

    correction_value = np.load('newdata_npy/correction_value.npy')

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
    #label_num = 4
    for j in range(number):
        start_time = time.time()
        #実装
        image = cv2.imread(f'../test/new/new_input2/input_{j}.png')
        #image = cv2.imread(f'test/simple/color_jpg/input_{j}.jpg')
        #image_c = np.float32(image) / 255
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('image.jpg', image_back)
        img = image.copy()
        #image_x, image_y, image_combine = edge_img(image)
        image_x, image_y, image_combine = edge_gauss(image)

        # 入力画像の特徴ベクトル
        results_x, res_x = matchtemp(image_x, new_eigenFaces_x)
        results_y, res_y = matchtemp(image_y, new_eigenFaces_y)
        # results = np.concatenate([results_x, results_y], 1)
        # results, res_x = matchtemp(image_combine, eigenFaces)
        w_e, h_e = new_eigenFaces_x[0].shape
        w1, h1 = res_x.shape
        w, h = results_x.shape
        end_time = time.time()

        start_time1 = time.time()
        F_x = np.transpose(results_x)
        F_y = np.transpose(results_y)
        #  # #カーネル
        # Ntrain = train_x.shape[0]
        # n_anchor = Ntrain
        # index = np.random.choice(Ntrain, n_anchor, replace=False)
        # X_anchor = train_x[index,:]
        # #ガウスカーネル
        # Vsigma = 0.3
        # features_k = np.exp(-sqdist(F, X_anchor) / (2*Vsigma*Vsigma))


        # 符号関数
        vH_i = np.sign(F_x)
        #vH_i = np.where(F_x > 0, 1, -1)
        vH_x = vH_i.astype(int)
        vH_i = np.sign(F_y)
        #vH_i = np.where(F_y > 0, 1, -1)
        vH_y = vH_i.astype(int)
        vH_i = np.concatenate([F_x, F_y], 1)
        vH = np.concatenate([vH_x, vH_y], 1)
        # #vH = vH.reshape(1, -1)
        end_time1 = time.time()

        classify = []
        posi_list = [{} for _ in range(label_num)]
        #count = 0
        start_time2 = time.time()
        #for i in range(h):       #ピクセル分繰り返す
        i = 0
        while i < h:    
            #1ピクセルごとにバイナリビット実行
            #features = np.array(features)
            # バイナリビットに変更する前
            #fea = F[i,:]
            feature = vH_i[i,:]
            # バイナリビットに変更後
            features = vH[i,:]

            n2, bits = X_train_seq.shape
            # ハミング距離の計算
            for n in range(n2):
                x = X_train_seq[n, :]
                #x = X_train_seq[n][:]
                #dis = hamming_dis(features, x)
                res = np.bitwise_xor(features, x)
                dis = np.count_nonzero(res)
                if dis == 0:
                    true = 1
                    cor_label = y_train_seq[0][n]
                    break
                else:
                    true = 0
                    cor_label = 4

            # 量子化誤差の閾値，適切に設定する必要有
            threshold = 0.6
            if true == 0:
                classify.append(4)
                #i += int(w_e/4)
                i += 1
            elif cor_label == 3:
                classify.append(3)
                #i += int(w_e/4)
                i += 1
                # continue
            else:
                classify.append(cor_label)
                error = quantization_error(feature, features)
                # 閾値以上のモノを除外
                if error <= threshold:
                    if cor_label == 0:
                        #count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 1:
                        #count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 2:
                        #count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                i += 1

        classify = np.array(classify)
        #print(count)
        #nms スコア量子化誤差
        overlap_thresh = 0.2
        location = []
        # # クラスごとのNMS
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

        # ラベルごとの補正係数
        correction_list = correction_value.tolist()
        label_weights = {index: value for index, value in enumerate(correction_list)}
        print(label_weights)
        # label_weights = {0: 0.5, 1: 1.0, 2: 1.5}

        final_boxes, final_scores, final_labels = nms_correction(all_boxes, all_scores, all_labels, overlap_thresh, label_weights)
        # final_boxes, final_labels = nms_all(all_boxes, all_scores, all_labels, overlap_thresh)

        # final_labels = all_labels[np.isin(all_boxes, final_boxes).all(axis=1)]

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
        # print("実行時間:", elapsed_time)
        # print(count)
        # print(location)
        #print(classify)
        cv2.imwrite(f'result/output/output_{j}.jpg', image)
        cv2.imwrite(f'result/out/out_{j}.jpg', img)
        # cv2.imshow("Img",img)
        # cv2.waitKey()

        # 正解位置
        loca = np.load('newdata_npy/new_location2.npy')
        #loca = np.load('npy/location_jpg.npy')
        correct = loca[j][:,:3]
        #correct = np.load('npy/placed_objects.npy')
        # 位置
        location = np.array(location)
        sort_index = np.lexsort((location[:,1], location[:,2]))
        prediction = location[sort_index]
        # 閾値
        accu_thre = 0.6
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
        total_distance += total_dis
        # print(correct)
        # print(prediction)
        #print(total_dis)
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
