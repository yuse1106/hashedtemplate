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

def rectangle_candidate(image, position, error, i, cor_label, scale):
    if scale == 1:
        x1 = int(i % h2)
        y1 = int(i / h2)
        x2 = int(x1 + W)
        y2 = int(y1 + H)
    else:
        x1 = int(scale*(i % h1))
        y1 = int(scale*(i / h1))
        x2 = int(x1 + w_e)
        y2 = int(y1 + h_e)
    position[error] = np.array([x1, y1, x2, y2])
    image = rectangle(image, cor_label, x1, y1, x2, y2)

    return image

def matchtemp(image, eigenFaces, scale):
    image_norm = np.float32(image) / 255.0
    results = []
    for eigenFace in eigenFaces:
        image_norm = cv2.resize(image_norm, dsize=(int(w_i/scale), int(h_i/scale)))
        eigenFace = cv2.resize(eigenFace, dsize=(int(w_e/scale), int(h_e/scale)))
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
   
    eigenFaces_x_21 = np.load('newdata_npy/new_eigenx_21.npy')
    eigenFaces_y_21 = np.load('newdata_npy/new_eigeny_21.npy')
    X_train_seq_21 = np.load('newdata_npy/X_train_seq_21.npy')
    y_train_seq_21 = np.load('newdata_npy/y_train_seq_21.npy')
    # eigenFaces_x_21 = np.load('newdata_npy/new_eigenx_42.npy')
    # eigenFaces_y_21 = np.load('newdata_npy/new_eigeny_42.npy')
    # X_train_seq_21 = np.load('newdata_npy/X_train_seq_42.npy')
    # y_train_seq_21 = np.load('newdata_npy/y_train_seq_42.npy')
    eigenFaces_x_84 = np.load('newdata_npy/new_eigenx_84.npy')
    eigenFaces_y_84 = np.load('newdata_npy/new_eigeny_84.npy')
    X_train_seq_84 = np.load('newdata_npy/X_train_seq_84.npy')
    y_train_seq_84 = np.load('newdata_npy/y_train_seq_84.npy')

    correction_value = np.load('newdata_npy/correction_value.npy')

    os.makedirs('result/new_harf_output', exist_ok=True)
    os.makedirs('result/new_search_output', exist_ok=True)
    os.makedirs('result/new_search_out', exist_ok=True)


    total_time = 0
    # total_time1 = 0
    # total_time2 = 0
    # total_time3 = 0
    total_loc_acc = 0
    total_acc = 0
    total_distance = 0
    number = 20 
    label_num = len(y_train_seq_21[0])
    #label_num = 4
    for j in range(number):
        start_time = time.time()
        #実装
        image = cv2.imread(f'../test/new/new_input2/input_{j}.png')
        #image = cv2.imread(f'../test/input_back1/input_{j}.jpg')
        #image_c = np.float32(image) / 255
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('image.jpg', image_back)
        img = image.copy()
        img_copy = image.copy()
        image_copy = image.copy()
        #image_x, image_y, image_combine = edge_img(image)
        image_x, image_y, image_combine = edge_gauss(image)
        w_i, h_i = image_x.shape
        # 保存
        # os.makedirs('new_input_edgex', exist_ok=True)
        # os.makedirs('new_input_edgey', exist_ok=True)
        # cv2.imwrite(f'new_input_edgex/input_{j}.jpg', image_x)
        # cv2.imwrite(f'new_input_edgey/input_{j}.jpg', image_y)

        # 疎の特徴ベクトル
        w_e, h_e = eigenFaces_x_21[0].shape
        scale = 4

        results_x, res_x = matchtemp(image_x, eigenFaces_x_21, scale)
        results_y, res_y = matchtemp(image_y, eigenFaces_y_21, scale)
        # results = np.concatenate([results_x, results_y], 1)
        # results, res_x = matchtemp(image_combine, eigenFaces)
        w1, h1 = res_x.shape
        w, h = results_x.shape

        F_x = np.transpose(results_x)
        F_y = np.transpose(results_y)
        #F = np.transpose(results)

        # vH_i = np.dot(F, trans)

        # 符号関数
        vH_i = np.sign(F_x)
        #vH_i = np.where(F_x > 0, 1, -1)
        vH_x = vH_i.astype(int)
        vH_i = np.sign(F_y)
        #vH_i = np.where(F_y > 0, 1, -1)
        vH_y = vH_i.astype(int)
        vH_i = np.concatenate([F_x, F_y], 1)
        vH = np.concatenate([vH_x, vH_y], 1)

        # 密の特徴ベクトル
         # 入力画像と比較
        original_scale = 1
        results_84 = []
        W, H = eigenFaces_x_84[0].shape[:2]
        results_x_84, res_x_84 = matchtemp(image_x, eigenFaces_x_84, original_scale)
        results_y_84, res_y_84 = matchtemp(image_y, eigenFaces_y_84, original_scale)
        w2, h2 = res_x_84.shape
        w_harf, h_harf = results_x_84.shape

        F_x_84 = np.transpose(results_x_84)
        F_y_84 = np.transpose(results_y_84)

        # 符号関数
        vH_i_84 = np.sign(F_x_84)
        #vH_i = np.where(F_x > 0, 1, -1)
        vH_x_84 = vH_i_84.astype(int)
        vH_i_84 = np.sign(F_y_84)
        #vH_i = np.where(F_y > 0, 1, -1)
        vH_y_84 = vH_i_84.astype(int)
        vH_i_84 = np.concatenate([F_x_84, F_y_84], 1)
        vH_84 = np.concatenate([vH_x_84, vH_y_84], 1)


        # np.savetxt('txt/vH.txt', vH, fmt='%d')

        classify = []
        posi_list = [{} for _ in range(label_num)]
        count = 0
        #for i in range(h):       #ピクセル分繰り返す
        i = 0
        # error = 10000
        while i < h:    
            #1ピクセルごとにバイナリビット実行
            #features = np.array(features)
            # バイナリビットに変更する前
            #fea = F[i,:]
            feature = vH_i[i,:]
            # バイナリビットに変更後
            features = vH[i,:]

            n2, bits = X_train_seq_21.shape

            for n in range(n2):
                x = X_train_seq_21[n, :]
                #x = X_train_seq[n][:]
                #dis = hamming_dis(features, x)
                #res = np.bitwise_xor(features, x)
                #dis = np.count_nonzero(res)
                dis = hamming_dis(features, x)
                if dis == 0:
                    true = 1
                    cor_label = y_train_seq_21[0][n]
                    break
                else:
                    true = 0
                    cor_label = 4


            threshold = 1.2
            if true == 0:
                classify.append(4)
                #i += int(w_e/4)
                i += 1
            elif cor_label == 3:
                classify.append(3)
                #i += int(w_e/4)
                i += 1
                # print(3)
                # continue
            else:
                classify.append(cor_label)
                error = quantization_error(feature, features)
                if error <= threshold:
                    if cor_label == 0:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label, scale)
                    elif cor_label == 1:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label, scale)
                    elif cor_label == 2:
                        count += 1
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label, scale)
                i += 1

        cv2.imwrite(f'result/new_harf_output/output_{j}.jpg', image)

        # 周辺ピクセル探索
        scale_posi_list = [{} for _ in range(label_num)]
        prediction1 = []
        search_pixel = 6
        for k, pos in enumerate(posi_list):
            boxes = np.array(list(pos.values()))
            for box in boxes:
                label = k
                x = box[0]
                y = box[1]
                error_min = 1000
                true = 0
                for m in range(search_pixel+1):
                    for l in range(search_pixel+1):
                        x1 = x + m - (int(search_pixel//2))
                        y1 = y + l - (int(search_pixel//2))
                        pixel = int(w2*y1+x1)
                        if pixel >= 110889:
                            break
                        feature1 = vH_i_84[pixel,:]
                        features1 = vH_84[pixel,:]

                        n2, bits = X_train_seq_84.shape

                        for n in range(n2):
                            x_tra = X_train_seq_84[n,:]
                            dis = hamming_dis(features1, x_tra)
                            if dis == 0:
                                cor_label = y_train_seq_84[0][n]
                                break
                            else:
                                cor_label = 4
                        threshold1 = 0.6
                        if label == cor_label:
                            error = quantization_error(feature1, features1)
                            if error < threshold1 and error < error_min :
                                error_min = error
                                pred_x = x1
                                pred_y = y1
                                final_pixel = int(w2*pred_y+pred_x)
                                pred_label = cor_label
                                true = 1
                if true == 1:
                    pred_list = [pred_label, pred_x, pred_y]
                    prediction1.append(pred_list)
                    if pred_label == 0:
                        img_copy = rectangle_candidate(img_copy, scale_posi_list[pred_label], error_min, final_pixel, pred_label, original_scale)
                        #img_copy = cv2.rectangle(img_copy, (pred_x, pred_y), (pred_x+W, pred_y+H), color=(0,0,255), thickness=4)
                    elif pred_label == 1:
                        img_copy = rectangle_candidate(img_copy, scale_posi_list[pred_label], error_min, final_pixel, pred_label, original_scale)
                        #img_copy = cv2.rectangle(img_copy, (pred_x, pred_y), (pred_x+W, pred_y+H), color=(50,255,255), thickness=4)
                    elif pred_label == 2:
                        img_copy = rectangle_candidate(img_copy, scale_posi_list[pred_label], error_min, final_pixel, pred_label, original_scale)
                        #img_copy = cv2.rectangle(img_copy, (pred_x, pred_y), (pred_x+W, pred_y+H), color=(50,200,255), thickness=4)

        prediction1 = np.array(prediction1)

        scale_location = []
        scale_overlap_thresh = 0.2

        # nmsをラベルごとに行わずに全候補に対して
        all_scores = []
        all_boxes = []
        all_labels = []

        for k, pos in enumerate(scale_posi_list):
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

        final_boxes, final_scores, final_labels = nms_correction(all_boxes, all_scores, all_labels, scale_overlap_thresh, label_weights)

        # final_boxes, final_labels = nms_all(all_boxes, all_scores, all_labels, overlap_thresh)

        # final_labels = all_labels[np.isin(all_boxes, final_boxes).all(axis=1)]

        image_copy, scale_location = draw_boxes(image_copy, final_boxes, final_labels, scale_location)

        end_time = time.time()
        #実行時間
        end_time2 = time.time()
        elapsed_time = end_time - start_time
        #elapsed_time1 = end_time1 - start_time1
        #elapsed_time2 = end_time2 - start_time2
        # print("実行時間:", elapsed_time, elapsed_time1, elapsed_time2)
        total_time += elapsed_time
        # total_time1 += elapsed_time
        # total_time2 += elapsed_time1
        # total_time3 += elapsed_time2
        # print("実行時間:", elapsed_time)
        # print(count)
        # print(location)
        #print(classify)
        #im = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f'result/new_search_output/output_{j}.jpg', img_copy)
        cv2.imwrite(f'result/new_search_out/out_{j}.jpg', image_copy)
        # cv2.imshow("Img",img)
        # cv2.waitKey()

        # 正解位置
        loca = np.load('newdata_npy/new_location2.npy')
        correct = loca[j][:,:3]
        #correct = np.load('npy/placed_objects.npy')
        # 位置
        scale_location = np.array(scale_location)
        sort_index = np.lexsort((scale_location[:,1], scale_location[:,2]))
        prediction2 = scale_location[sort_index]
        # 閾値
        accu_thre = 0.55
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction2, accu_thre, distance_thre)
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
    # average_time1 = round(total_time1 / number, 3)
    # average_time2 = round(total_time2 / number, 3)
    # average_time3 = round(total_time3 / number, 3)
    average_pixel = round(total_distance / number, 3)
    average_loc_acc = round(total_loc_acc / number, 3)
    average_acc = round(total_acc / number, 3)
    print('平均時間：', average_time)
    print('平均ピクセル誤差：', average_pixel)
    print('平均位置精度：', average_loc_acc)
    print('平均精度：', average_acc)
