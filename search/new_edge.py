import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import cv2
import glob
import numpy as np
import os
import time
from skimage.exposure import rescale_intensity
from utils import *

# def get_data_edge(path):
#     path_list = glob.glob(path+'/*')
#     sobel_x = []
#     sobel_y = []
#     canny_list = []
#     y_train = []
#     for label, pic_path in enumerate(path_list):
#         train_list = glob.glob(pic_path+'/*.png')
#         train_list = sorted(train_list, key=lambda x: int(''.join(filter(str.isdigit,x))))
#         dir_path = 'newdata/sobel_x'
#         os.makedirs(dir_path+f'/newdata{label}', exist_ok=True)
#         dir_path1 = 'newdata/sobel_y'
#         os.makedirs(dir_path1+f'/newdata{label}', exist_ok=True)
#         dir_path2 = 'newdata/sobel_combine'
#         os.makedirs(dir_path2+f'/newdata{label}', exist_ok=True)
#         for i, train in enumerate(train_list):
#             x_train = cv2.imread(train)
#             # エッジ検出
#             #sobel_x_img, sobel_y_img, canny = edge_img(x_train)
#             sobel_x_img, sobel_y_img, canny = edge_gauss(x_train)

#             sobel_x_img1 = rescale_intensity(sobel_x_img, out_range=(0,255))
#             sobel_x_img1 = np.dstack([sobel_x_img1.astype("uint8")])
#             sobel_y_img1 = rescale_intensity(sobel_y_img, out_range=(0,255))
#             sobel_y_img1 = np.dstack([sobel_y_img1.astype("uint8")])
#             canny1 = rescale_intensity(canny, out_range=(0,255))
#             canny1 = np.dstack([canny1.astype("uint8")])
#             if label == 0:
#                 cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
#                 cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
#                 cv2.imwrite(dir_path2+f'/newdata{label}/canny_{i}.jpg', canny1)
#             elif label == 1:
#                 cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
#                 cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
#                 cv2.imwrite(dir_path2+f'newdata{label}/canny_{i}.jpg', canny1)
#             elif label == 2:
#                 cv2.imwrite(dir_path+f'/newdata{label}/sobelx_{i}.jpg', sobel_x_img1)
#                 cv2.imwrite(dir_path1+f'/newdata{label}/sobely_{i}.jpg', sobel_y_img1)
#                 cv2.imwrite(dir_path2+f'newdata{label}/canny_{i}.jpg', canny1)

#             sobel_x_img = np.float32(sobel_x_img) / 255.0
#             sobel_y_img = np.float32(sobel_y_img) / 255.0
#             canny = np.float32(canny) / 255.0
#             sobel_x.append(sobel_x_img)
#             sobel_y.append(sobel_y_img)
#             canny_list.append(canny)
        
#         y_train += [label]*len(train_list)
#     return sobel_x, sobel_y, canny_list, y_train, label

def template_edge(mean, eigenVectors, i, size):
    eigen = []
    dir = f'eigenFace'
    os.makedirs(dir+f'/newdata{i}', exist_ok=True)
    for j, eigenVector in enumerate(eigenVectors):
        eigenFace = eigenVector.reshape(size)
        eigenFace1 = rescale_intensity(eigenFace, out_range=(0,255))
        eigenFace1 = np.dstack([eigenFace1.astype("uint8")])
        cv2.imwrite(dir+f'/newdata{i}/eigen_{j}.jpg', eigenFace1)
        eigen.append(eigenFace)

    return eigen

def create_eigen_edge(train, NUM_EIGEN, size):
    eigenFaces = []
    concat_eigenVectors = []
    for i in range(label):
        data = createMatrix(train[i*360:(i+1)*360], size)
        mean, eigenVectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN)
        eigen = template_edge(mean, eigenVectors, i, size)
        eigenFaces.append(eigen)
        concat_eigenVectors.append(eigenVectors)
    eigenFaces = np.vstack(eigenFaces)
    #eigenVectors = np.concatenate(eigenVectors)
    eigenVectors = np.vstack(concat_eigenVectors)

    return eigenFaces, eigenVectors 

if __name__ == '__main__':
    start_time = time.time()
    # テンプレート数
    NUM_EIGEN_TEMPS_EDGE = 150
    # ファイルディレクトリ
    path = "../new_dataset"
    # ラベルとエッジ強度の辞書
    label_edge = {}
    edge_x, edge_y, edge, y_train, label, label_edge = get_data_edge(path, label_edge)
    y_train = np.array(y_train)
    #sobel = (sobel_x + sobel_y) / 2
    os.makedirs('newdata_npy', exist_ok=True)
    np.save('newdata_npy/sobel_x.npy', edge_x)
    np.save('newdata_npy/sobel_y.npy', edge_y)
    # np.save('newdata_npy/canny_list.npy', canny_list)
    np.save('newdata_npy/y_train.npy', y_train)

    # 補正係数
    correction_value = correction_factor(label_edge)

    size_edge = edge_x[0].shape

    eigenFaces_x, eigenVectors_x = create_eigen_edge(edge_x, NUM_EIGEN_TEMPS_EDGE, size_edge)
    eigenFaces_y, eigenVectors_y = create_eigen_edge(edge_y, NUM_EIGEN_TEMPS_EDGE, size_edge)
    # canny, cannyVectors = create_eigen_edge(canny_list, NUM_EIGEN_TEMPS_EDGE, size_edge)
    # print(eigenVectors_x.shape)
    # print(eigenVectors_y.shape)

    #eigenFaces = (eigenFaces_x + eigenFaces_y) /2
    #eigenVectors = (eigenVectors_x + eigenVectors_y) / 2

    print(correction_value)

    np.save('newdata_npy/correction_value.npy', correction_value)
    print("eigenFaceの大きさ:", eigenFaces_x.shape, eigenFaces_y.shape)
    #np.save('newdata_npy/eigenFaces.npy', eigenFaces)
    np.save('newdata_npy/eigenFaces_x.npy', eigenFaces_x)
    np.save('newdata_npy/eigenFaces_y.npy', eigenFaces_y)
    #np.save('newdata_npy/eigenVectors.npy', eigenVectors)
    np.save('newdata_npy/eigenVectors_x.npy', eigenVectors_x)
    np.save('newdata_npy/eigenVectors_y.npy', eigenVectors_y)
    end_time = time.time()
    time_calculation(start_time, end_time)
    print(len(edge_x))