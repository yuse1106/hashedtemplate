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


# 候補領域を描画する
def rectangle_candidate(image, position, error, i, cor_label):
    # count += 1
    x1 = int(i % h1)
    y1 = int(i / h1)
    x2 = x1 + w_e
    y2 = y1 + h_e
    position[error] = np.array([x1, y1, x2, y2])
    image = rectangle(image, cor_label, x1, y1, x2, y2)

    return image

# バウンディングボックス
def draw_boxes(img, boxes, i, location):
    dst = img.copy()
    for x1, y1, x2, y2 in boxes:
        location.append((i, x1, y1))
        dst = rectangle(dst, i, x1, y1, x2, y2)
        
    return dst, location


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

    eigenFaces = np.load('npy/eigenFaces.npy')
    train_x = np.load('npy/train_x.npy')
    y_train = np.load('npy/y_train.npy')
    X_train_seq = np.load('npy/X_train_seq.npy')
    y_train_seq = np.load('npy/y_train_seq.npy')
    trans = np.load('npy/trans.npy')

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
        #実装
        image = cv2.imread(f'../test/simple/color_jpg/input_{j}.jpg')
        #image = cv2.imread(f'test/simple/color_jpg_back/input_{j}.jpg')
        image_c = np.float32(image) / 255
        #cv2.imwrite('image.jpg', image_back)
        img = image.copy()
        #img = image_back.copy()
        results = []
        for eigenFace in eigenFaces:
            res = cv2.matchTemplate(image_c, eigenFace, cv2.TM_CCORR)
            # res = cv2.matchTemplate(image_c, eigenFace, cv2.TM_CCOEFF)
            result = res.flatten()
            results.append(result)
        w_e, h_e = eigenFace.shape[:2]
        results = np.array(results)
        w1, h1 = res.shape
        w, h = results.shape
        end_time =time.time()

        start_time1 = time.time()
        F = np.transpose(results)
        #  # #カーネル
        # Ntrain = train_x.shape[0]
        # n_anchor = Ntrain
        # index = np.random.choice(Ntrain, n_anchor, replace=False)
        # X_anchor = train_x[index,:]
        # #ガウスカーネル
        # Vsigma = 0.3
        # features_k = np.exp(-sqdist(F, X_anchor) / (2*Vsigma*Vsigma))

        #バイナリビット生成
        vH_i = np.dot(F, trans) #>0

        np.save('npy/vH_i.npy', vH_i)

        # 符号関数, -1or0or1で表される
        vH = np.sign(vH_i)
        vH = vH.astype(int)

        np.save('npy/vH.npy', vH)
        end_time1 = time.time()

        classify = []
        posi_list = [{} for _ in range(label_num-1)]
        print(posi_list)
        # posi = []
        count = 0
        #start_time = time.time()
        i = 0
        start_time2 =time.time()
        while i < h:
            
            #1ピクセルごとにバイナリビット実行
            fea = F[i, :]
            # バイナリビットに変更する前，１かー１に近づいている
            feature = vH_i[i,:]
            # バイナリビットに変更後
            features = vH[i,:]

            n2, bits = X_train_seq.shape

            # ハッシュテーブルのバイナリコードと比較する，ハミング距離が０になればその値を割り当てる
            for n in range(n2):
                x = X_train_seq[n, :]
                dis = hamming_dis(features, x)
                if dis == 0:
                    true = 1
                    cor_label = y_train_seq[0][n]
                    break
                else:
                    true = 0
                    cor_label = 4

            # 閾値処理，num_bitのビット数によって適切な値を設定する必要がある
            threshold = 0.1
            if true == 0:
                classify.append(4+4)
                i += 1
                # print('0')
            elif cor_label == 3:
                classify.append(cor_label+4)
                i += 1
            else:
                #cor_label = y_train_seq[ham_sort_index[0]]
                classify.append(cor_label+4)
                error = quantization_error(feature, features)
                if error <= threshold:
                    if cor_label == 0:
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 1:
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                    elif cor_label == 2:
                        image = rectangle_candidate(image, posi_list[cor_label], error, i, cor_label)
                i += 1

        classify = np.array(classify)
        np.save('npy/classify.npy', classify)

        #nms スコア量子化誤差
        # posi.append(apple_posi)
        # posi.append(lemon_posi)
        # posi.append(orange_posi)
        location = []
        overlap_thresh = 0.5
        for k, pos in enumerate(posi_list):
            scores = np.array(list(pos.keys()))
            boxes = np.array(list(pos.values()))
            box, score = nms(boxes, scores, overlap_thresh)
            img, location = draw_boxes(img, box, k, location)
            #img, location = draw_boxespose(img, box, i, location)


        #実行時間
        end_time2 = time.time()
        elapsed_time = end_time - start_time
        elapsed_time1 = end_time1 - start_time1
        elapsed_time2 = end_time2 - start_time2
        print(f"実行時間: {elapsed_time:.3f}, {elapsed_time1:.3f}, {elapsed_time2:.3f}")
        total_time += elapsed_time + elapsed_time1 + elapsed_time2
        total_time1 += elapsed_time
        total_time2 += elapsed_time1
        total_time3 += elapsed_time2
        cv2.imwrite('out.jpg', image)
        cv2.imwrite(f'result/output/output_{j}.jpg', image)
        cv2.imwrite(f'result/out/out_{j}.jpg', img)

        # 正解位置
        loca = np.load('npy/location_jpg.npy')
        correct = loca[j][:,:3]
        # 位置
        location = np.array(location)
        sort_index = np.lexsort((location[:,1], location[:,2]))
        prediction = location[sort_index]
        # 閾値
        accu_thre = 0.6
        distance_thre = 5
        accuracy_location, accuracy_label, dis, total_dis = accuracy(correct, prediction, accu_thre, distance_thre)
        total_distance += total_dis
        print(dis)
        # print("位置の正解率:", accuracy_location)
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
