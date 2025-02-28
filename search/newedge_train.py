import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
import glob
import time
from skimage.exposure import rescale_intensity
from hadamard import hadamard2
from utils import *

#テンプレートマッチング
def template(eigenVectors):
    # Create a container to hold eigen faces.
    eigenFaces = []
    # Reshape eigen vectors to eigen faces.
    for i, eigenVector in enumerate(eigenVectors):
        eigenFace = eigenVector.reshape(size)  
        eigenFace1 = rescale_intensity(eigenFace, out_range=(0,255))   
        eigenFace1 = np.dstack([eigenFace1.astype("uint8")])
        # cv2.imshow("com", eigenFace1)
        # cv2.waitKey(0)
        # cv2.imread()
        os.makedirs('new_eigen', exist_ok=True)
        cv2.imwrite(f'new_eigen/eigen_{i}.jpg', eigenFace1)
        eigenFaces.append(eigenFace)
    return eigenFaces

def train_data(sobel, eigenFaces, scale):
    train_x = np.ones((len(sobel), len(eigenFaces)))
    for i, img in enumerate(sobel):
        for j, eigenFace in enumerate(eigenFaces):
            img = cv2.resize(img, dsize=(int(w/scale), int(h/scale)))
            eigenFace = cv2.resize(eigenFace, dsize=(int(w1/scale), int(h1/scale)))
            result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCORR)
            train_x[i,j] = result

    return train_x

def new_eigen(train, y_train, eigenVectors, num_bit):
    Fmap = {'nu':1e-5, 'lambda':1e-2}
    trans = hadamard2(train, y_train, num_bit, Fmap)
    trans = trans.astype(np.float32)

    new_trans = np.transpose(trans)
    new_eigenVectors = np.dot(new_trans, eigenVectors)

    new_eigenFaces = template(new_eigenVectors)

    return new_eigenFaces

if __name__ == '__main__':
    start_time = time.time()

    edge_x = np.load('newdata_npy/sobel_x.npy')
    edge_y = np.load('newdata_npy/sobel_y.npy')
    y_train = np.load('newdata_npy/y_train.npy')
    eigenFaces_x = np.load('newdata_npy/eigenFaces_x.npy')
    eigenFaces_y = np.load('newdata_npy/eigenFaces_y.npy')
    eigenVectors_x = np.load('newdata_npy/eigenVectors_x.npy')
    eigenVectors_y = np.load('newdata_npy/eigenVectors_y.npy')

    # # 訓練データにデータを加える
    # path = "../add_dataset/new_add30"
    # edge_x, edge_y, y_train = add_edge_data(path, edge_x, edge_y, y_train)

    size = edge_x[0].shape
    w, h= edge_x[0].shape
    w1, h1 = eigenFaces_x[0].shape
    scale = 1

    #eigenVectors = np.concatenate([eigenVectors_x, eigenVectors_y])

    train_data_x = train_data(edge_x, eigenFaces_x, scale)
    train_data_y = train_data(edge_y, eigenFaces_y, scale)
    train_data_x = np.array(train_data_x)
    train_data_y = np.array(train_data_y)
    #train_data_x = train_data(sobel_x, sobel_x)
    #train_data_y = train_data(sobel_y, sobel_y)
    # train_x = np.concatenate([train_data_x, train_data_y], 1)

    # train_x = train_data(sobel, eigenFaces)
    
    
    # 状態変換行列（ハッシュ関数生成）
    num_bit = 4

    # new_eigenFaces = new_eigen(train_x, y_train, sobelVectors, num_bit)

    new_eigenFaces_x = new_eigen(train_data_x, y_train, eigenVectors_x, num_bit)

    new_eigenFaces_y = new_eigen(train_data_y, y_train, eigenVectors_y, num_bit)

    # #学習データ
    # new_train_x = np.ones((len(x_train), len(new_eigenFaces)))
    # for i,img in enumerate(x_train):
    #     for j,new_eigenFace in enumerate(new_eigenFaces):
    #         result = cv2.matchTemplate(img, new_eigenFace, cv2.TM_CCORR)
    #         new_train_x[i,j]=result

    new_train_data_x = train_data(edge_x, new_eigenFaces_x, scale)
    new_train_data_y = train_data(edge_y, new_eigenFaces_y, scale)
    # new_train_x = np.concatenate([new_train_data_x, new_train_data_y], 1)

    # 0以上を1，それ以外を-1にする
    # X_train = np.where(new_train_x > 0, 1, -1)
    # X_train = X_train.astype(int)
    # signによる識別関数
    X_train_x = np.sign(new_train_data_x)
    #X_train_x = np.where(new_train_data_x >0, 1, -1)
    X_train_x = X_train_x.astype(int)
    X_train_y = np.sign(new_train_data_y)
    #X_train_y = np.where(new_train_data_y >0, 1, -1)
    X_train_y = X_train_y.astype(int)
    X_train = np.concatenate([X_train_x, X_train_y], 1)
    

    # バイナリビットに変換しないときの最初の特徴ベクトルを取得
    X_train_seq = []
    # for j in range(0, 1081, 360):
    #     X_train_seq.append(X_train[j])
    # 変換したとき重複ないように
    X_train_seq, indice = np.unique(X_train, axis = 0, return_index = True)
    # X_train_seq = list(dict.fromkeys(map(tuple, X_train)))
    X_train_seq = np.array(X_train_seq)
    print(X_train_seq)
    y_train_seq = []
    # for j in range(0, 1080, 360):
    #     y_train_seq.append(y_train[j])
    y = y_train[indice]
    y_train_seq.append(y)
    #y_train_seq = np.unique(y_train, axis=0)
    print(y_train_seq)
    #X_train = X_train_i.astype(int)
    #np.savetxt('txt/my_data.txt', X_train, fmt='%d')

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")

    if scale == 1:
        np.save('newdata_npy/new_eigenx_84.npy', new_eigenFaces_x)
        np.save('newdata_npy/new_eigeny_84.npy', new_eigenFaces_y)
        np.save('newdata_npy/X_train_seq_84.npy', X_train_seq)
        np.save('newdata_npy/y_train_seq_84.npy', y_train_seq)
    elif scale == 2:
        np.save('newdata_npy/new_eigenx_42.npy', new_eigenFaces_x)
        np.save('newdata_npy/new_eigeny_42.npy', new_eigenFaces_y)
        np.save('newdata_npy/X_train_seq_42.npy', X_train_seq)
        np.save('newdata_npy/y_train_seq_42.npy', y_train_seq)
    elif scale == 4:
        np.save('newdata_npy/new_eigenx_21.npy', new_eigenFaces_x)
        np.save('newdata_npy/new_eigeny_21.npy', new_eigenFaces_y)
        np.save('newdata_npy/X_train_seq_21.npy', X_train_seq)
        np.save('newdata_npy/y_train_seq_21.npy', y_train_seq)