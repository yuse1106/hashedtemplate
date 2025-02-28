import numpy as np
import cv2
import glob
import math
import os
from skimage.exposure import rescale_intensity

# ２乗距離
def sqdist(X, Y):
    distance = np.sum((X[:, np.newaxis]-Y) ** 2, axis=-1)
    return distance

# 画像を一列にして行列を生成する
def createMatrix(images, size):
    print("Creating data matrix", end = "...\n")
    images_num = len(images)
    # size = images[0].shape
    #　カラー画像とグレー画像で入れ替え
    if len(size) == 2:
        data = np.zeros((images_num, size[0]*size[1]), dtype = np.float32)
    else:
        data = np.zeros((images_num, size[0]*size[1]*size[2]), dtype = np.float32)
    #data = np.zeros((images_num, size[0]*size[1]), dtype = np.float32)
    for i in range(images_num):
        #image = images[i].flatten()
        image = images[i].reshape(-1)
        data[i,:]= image

    return data

#画像データセット作成
def get_data(path, file):
    path_list = glob.glob(path+'/*')
    
    train_x = []
    train_y = []
    for label, pic_path in enumerate(path_list):
        #pic_path = glob.glob(path + "/" + pic_path)
        if file == 0:
            train_list = glob.glob(pic_path+'/*.jpg')
        else:
            train_list = glob.glob(pic_path+'/*.png')
        train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
        for i in train_list:
            x_train = cv2.imread(i)
            # x_train = cv2.cvtColor(x_train, cv2.COLOR_BGR2GRAY)
            #画像サイズ変更
            x_train_c = cv2.resize(x_train, (84,84))
            if x_train_c is None:
                print("image:{} not read properly".format(i))
            else:
                #正規化
                x_train = np.float32(x_train_c) / 255
                #リストに追加
                train_x.append(x_train)
        train_y += [label]*len(train_list)

    return train_x, train_y, label

def get_data_edge(path, label_edge):
    path_list = glob.glob(path+'/*')
    sobel_x = []
    sobel_y = []
    canny_list = []
    y_train = []
    for label, pic_path in enumerate(path_list):
        train_list = glob.glob(pic_path+'/*')
        train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
        dir_path = 'sobel_x'
        os.makedirs(dir_path+f'/newdata{label}', exist_ok=True)
        dir_path1 = 'sobel_y'
        os.makedirs(dir_path1+f'/newdata{label}', exist_ok=True)
        # dir_path2 = 'newdata/sobel_combine'
        # os.makedirs(dir_path2+f'/newdata{label}', exist_ok=True)
        for i, train in enumerate(train_list):
            x_train = cv2.imread(train)
            # エッジ検出
            #sobel_x_img, sobel_y_img, canny = edge_img(x_train)
            sobel_x_img, sobel_y_img, canny = edge_gauss(x_train)

            # エッジ強度の計算
            magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)
            average_magnitude = np.mean(magnitude)
            #print(average_magnitude)

            if label in label_edge:
                label_edge[label] += average_magnitude / 360
            else:
                label_edge[label] = average_magnitude / 360

            # # 画像として保存できるように
            # sobel_x_img1 = rescale_intensity(sobel_x_img, out_range=(0,255))
            # sobel_x_img1 = np.dstack([sobel_x_img1.astype("uint8")])
            # sobel_y_img1 = rescale_intensity(sobel_y_img, out_range=(0,255))
            # sobel_y_img1 = np.dstack([sobel_y_img1.astype("uint8")])
            # canny1 = rescale_intensity(canny, out_range=(0,255))
            # canny1 = np.dstack([canny1.astype("uint8")])
            # if label == 0:
            #     cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
            #     cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
            #     cv2.imwrite(dir_path2+f'/newdata{label}/canny_{i}.jpg', canny1)
            # elif label == 1:
            #     cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
            #     cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
            #     cv2.imwrite(dir_path2+f'newdata{label}/canny_{i}.jpg', canny1)
            # elif label == 2:
            #     cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
            #     cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
            #     cv2.imwrite(dir_path2+f'newdata{label}/canny_{i}.jpg', canny1)

            sobel_x_img = np.float32(sobel_x_img) / 255.0
            sobel_y_img = np.float32(sobel_y_img) / 255.0
            canny = np.float32(canny) / 255.0
            sobel_x.append(sobel_x_img)
            sobel_y.append(sobel_y_img)
            canny_list.append(canny)
        
        y_train += [label]*len(train_list)
    return sobel_x, sobel_y, canny_list, y_train, label, label_edge

# ハミング距離
def hamming_dis(V, x):
    # ビット列をNumPyの配列に変換
    xor_result = np.bitwise_xor(V,x)
    dis = np.count_nonzero(xor_result)
    return dis

# 量子化誤差
def quantization_error(feature, binary_feature):
    dis = np.sqrt(np.sum((feature-binary_feature) ** 2))
    return dis

# ラベルごとのバウンディングヴボックス
def rectangle(image, i, x1, y1, x2, y2):
    if i == 0:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
    elif i ==1:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(50, 255, 255), thickness=4)
    elif i ==2:
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(50, 200, 255), thickness=4)

    return image


# nms
def nms_all(boxes, scores, labels, overlap_thresh):
    if len(boxes) <= 1:
        return boxes

    # float 型に変換する。
    #boxes = boxes.astype("float")

    # (NumBoxes, 4) の numpy 配列を x1, y1, x2, y2 の一覧を表す4つの (NumBoxes, 1) の numpy 配列に分割する。
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))

    # 矩形の面積を計算する。
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(-scores)  # スコアをソートした一覧
    selected = []  # NMS により選択された一覧

    # indices がなくなるまでループする。
    while len(indices) > 0:
        # 残っている中で最もスコアが高い。
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算
        # overlap = (i_w * i_h) / area[remaining_indices]
        overlap = (i_w * i_h) / (2 * area[remaining_indices] - i_w * i_h)

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # 選択された短形の一覧を返す。
    return boxes[selected].astype("int"), labels[selected].astype("int")

def nms_general(boxes, scores, labels, overlap_thresh):
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
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        # overlap = (i_w * i_h) / area[remaining_indices]
        overlap = (i_w * i_h) / (2 * area[remaining_indices] - i_w * i_h)

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # 選択された短形の一覧を返す。
    return selected

# 補正係数を適用した場合のNMS
def nms_correction(boxes, scores, labels, overlap_thresh, label_weights):
    if len(boxes) <= 1:
        return boxes, scores, labels

    # ボックスを分割
    x1, y1, x2, y2 = np.squeeze(np.split(boxes, 4, axis=1))
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(-scores)  # スコア降順
    selected = []  # 最終的に選択されたインデックス
    deferred_indices = []  # 再度NMSを適用するインデックス

    while len(indices) > 0:
        deferred_indices = []
        last = len(indices) - 1
        selected_index = indices[last]
        remaining_indices = indices[:last]
        # selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分を計算
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 共通部分の幅・高さを計算
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # Overlap Ratioを計算
        overlap = (i_w * i_h) / (area[selected_index] + area[remaining_indices] - i_w * i_h)

        # ラベルが異なりOverlapが閾値以上の場合
        differing_labels = labels[remaining_indices] != labels[selected_index]
        high_overlap = overlap > overlap_thresh
        to_defer = np.where(differing_labels & high_overlap)[0]
        # indices = np.delete(indices, np.concatenate(([last], to_defer)))

        if len(to_defer) > 0:
            # 再処理するインデックスに保存
            deferred_indices.extend(remaining_indices[to_defer])
            deferred_indices.append(selected_index)

            # to_remove = np.where(differing_labels & high_overlap)[0]
            # to_remove = to_remove[to_remove < len(indices)]
            # indices = np.delete(indices, to_remove)
        else:
            selected.append(selected_index)
        
        # Overlapが閾値以上かつラベルが同じ場合はindicesから削除
        same_labels = labels[remaining_indices] == labels[selected_index]
        to_remove_same = np.where(high_overlap & same_labels)[0]
        indices = np.delete(indices, np.concatenate(([last], to_remove_same)))

        #to_remove_diff = np.where(high_overlap & differing_labels)[0]
        #indices = np.delete(indices, to_remove_diff)
        indices = indices[~np.isin(indices, deferred_indices)]
    
    # 再度NMSを deferred_indices に適用
        if len(deferred_indices) > 0:
            deferred_boxes = boxes[deferred_indices]
            deferred_scores = scores[deferred_indices]
            deferred_labels = labels[deferred_indices]

            # ラベルごとにスコア補正
            for i, label in enumerate(deferred_labels):
                deferred_scores[i] *= label_weights.get(label, 1.0)

            selected_deferred = nms_general(deferred_boxes, deferred_scores, deferred_labels, overlap_thresh)

            selected.extend(np.array(deferred_indices)[selected_deferred])

    return boxes[selected], scores[selected], labels[selected]

# nms
def nms(boxes, scores, overlap_thresh):
    if len(boxes) <= 1:
        return boxes, scores

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
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        # 選択した短形と残りの短形の共通部分の x1, y1, x2, y2 を計算する。
        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        # 選択した短形と残りの短形の共通部分の幅及び高さを計算する。
        # 共通部分がない場合は、幅や高さは負の値になるので、その場合、幅や高さは 0 とする。
        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        # 選択した短形と残りの短形の Overlap Ratio を計算する。
        # overlap = (i_w * i_h) / area[remaining_indices]
        overlap = (i_w * i_h) / (2 * area[remaining_indices] - i_w * i_h)

        # 選択した短形及び OVerlap Ratio が閾値以上の短形を indices から削除する。
        indices = np.delete(
            indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
        )

    # 選択された短形の一覧を返す。
    return boxes[selected].astype("int"), scores[selected].astype("float")

# ガウスカーネル, カーネルトリックに使用
def gaussian_kernel(X, X_anchor, sigma):
     kernel = np.exp(-sqdist(X, X_anchor) / (2*sigma*sigma))
     return kernel

# エッジ抽出，ガウスカーネルをしてsobelフィルタ適用
def edge_img(gray_img):
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    sigma = 1
    kernel_size = 5
    blur = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), sigma)

    # canny = cv2.Canny(blur, threshold1=30, threshold2=100)

    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    #sobel_x = cv2.convertScaleAbs(sobel_x)
    #sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    return sobel_x, sobel_y, sobel_combined

def edge_gauss(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sigma = 3.0
    N = int(np.floor(sigma*3.0)*2+1)
    h = np.zeros((N, N), dtype=np.float32)
    hx = np.zeros_like(h)
    hy = np.zeros_like(h)
    for sy in range(N):
        for sx in range(N):
            x = sx - (N-1) / 2
            y = sy - (N-1) /2
            gx = -x * np.exp(-(x*x+y*y) / (2.0*sigma*sigma)) / (2*math.pi*sigma*sigma*sigma*sigma)
            hx[sy, sx] = gx
            gy = -y * np.exp(-(x*x+y*y) / (2.0*sigma*sigma)) / (2*math.pi*sigma*sigma*sigma*sigma)
            hy[sy,sx] = gy
    dx = cv2.filter2D(img, cv2.CV_32F, hx)
    dy = cv2.filter2D(img, cv2.CV_32F, hy)

    combined = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

    return dx, dy, combined

# 補正係数
def correction_factor(label_edge):
    value_list = [round(value, 2) for value in label_edge.values()]
    list_max = max(value_list)

    sorted_value = sorted(value_list)
    sorted_indices = np.argsort(value_list)[::-1]
    # 最大値で割る
    normalized_value = [1.5 - (value / list_max) for value in sorted_value]
    adjusted_value = [None] * len(value_list)

    for i, idx in enumerate(sorted_indices):
        adjusted_value[idx] = normalized_value[-(i+1)] * (i*i+1)
        #adjusted_value[idx] = normalized_value[i] * (i+1)

    normalized_array = np.array(adjusted_value)

    return normalized_array
    

def time_calculation(start, end):
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")

def add_data(path, train_x, train_y):
    path_list = glob.glob(path+'/*')

    for label, pic_path in enumerate(path_list):
        train_list = glob.glob(pic_path+'/*')
        train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
        for i in train_list:
            x_train = cv2.imread(i)
            # 正規化
            x_train = np.float32(x_train) / 255
            # リストに追加
            # train_x.append(x_train)
            train_x = np.concatenate([train_x, np.expand_dims(x_train, axis=0)], axis=0)
        train_y = np.concatenate([train_y, np.full(len(train_list), label)])
        # train_y += [label]*len(train_list)
    
    return train_x, train_y

def add_edge_data(path, edge_x, edge_y, train_y):
    path_list = glob.glob(path+'/*')
    os.makedirs('newdata/edge/class1', exist_ok=True)
    os.makedirs('newdata/edge/class2', exist_ok=True)
    os.makedirs('newdata/edge/class3', exist_ok=True)

    for label, pic_path in enumerate(path_list):
        train_list = glob.glob(pic_path+'/*')
        # train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit, x))))
        for i in train_list:
            x_train = cv2.imread(i)
            # x_train = np.float32(x_train) / 255.0
            edge_x_img, edge_y_img, edge = edge_gauss(x_train)
            edge_x_img1 = rescale_intensity(edge_x_img, out_range=(0,255))
            edge_x_img1 = np.dstack([edge_x_img1.astype('uint8')])
            if label == 0:
                cv2.imwrite(f'newdata/edge/class1/edge_{i}.jpg', edge_x_img1)
            elif label == 1:
                cv2.imwrite(f'newdata/edge/class2/edge_{i}.jpg', edge_x_img1)
            elif label == 2:
                cv2.imwrite(f'newdata/edge/class3/edge_{i}.jpg', edge_x_img1)

            edge_x_img = np.float32(edge_x_img) / 255.0
            edge_y_img = np.float32(edge_y_img) / 255.0

            edge_x = np.concatenate([edge_x, np.expand_dims(edge_x_img, axis=0)], axis=0)
            edge_y = np.concatenate([edge_y, np.expand_dims(edge_y_img, axis=0)], axis=0)

        train_y = np.concatenate([train_y, np.full(len(train_list), label)])

    return edge_x, edge_y, train_y
