import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import cv2
import glob
from hadamard import hadamard2
import time
from utils import *

if __name__ == '__main__':
    start_time = time.time()
    x_train = np.load('npy/x_train.npy')
    eigenFaces = np.load('npy/eigenFaces.npy')
    y_train = np.load('npy/y_train.npy')

    # # 訓練データにデータを加える
    # path = "../add_dataset/add_train"
    # x_train, y_train = add_data(path, x_train, y_train)
    # print(x_train.shape, len(y_train))

    #学習データ
    train_x = np.ones((len(x_train), len(eigenFaces)))
    for i,img in enumerate(x_train):
        for j,eigenFace in enumerate(eigenFaces):
            result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCORR)
            # result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCOEFF)
            train_x[i,j]=result

    #np.save('npy/train_x.npy', train_x)

    # #カーネル
    # Ntrain = train_x.shape[0]
    # n_anchor = Ntrain
    # index = np.random.choice(Ntrain, n_anchor, replace=False)
    # X_anchor = train_x[index,:]
    # #ガウスカーネル
    # Vsigma = 0.3
    # train_X = np.exp(-sqdist(train_x, X_anchor) / (2*Vsigma*Vsigma))

    num_bit = 4
    #状態変換行列生成，アダマール行列によりnum_bitで指定した数の次元に削減
    Fmap = {'nu': 1e-5, 'lambda': 1e-2}
    trans = hadamard2(train_x, y_train, num_bit, Fmap)
    trans = trans.astype(np.float32)
    #学習画像データと状態変換行列
    X_train_i = np.dot(train_x, trans) #>0
    #np.save('npy/X_train_i.npy', X_train_i)


    # 0以上を1，それ以外を-1にする
    # X_train = X_train_i >0
    # X_train = X_train.astype(int)
    # signによる識別関数，すべてのデータ
    X_train = np.sign(X_train_i)
    X_train = X_train.astype(int)

    #np.save('npy/X_train_bit.npy', X_train)

    # 重複をなくす，理想としては，１つのラベルに対して１つのバイナリコードが生成される
    X_train_seq = []
    # for j in range(0, 1081, 360):
    #     X_train_seq.append(X_train[j])
    X_train_seq, indice = np.unique(X_train, axis = 0, return_index = True)
    X_train_seq = np.array(X_train_seq)
    print(X_train_seq)
    # 変換したとき重複ないように
    y_train_seq = []
    # for j in range(0, 1080, 360):
    #     y_train_seq.append(y_train[j])
    # print(y_train_seq)
    y = y_train[indice]
    y_train_seq.append(y)
    print(y_train_seq)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")

    np.save('npy/trans.npy', trans)
    np.save('npy/X_train_seq.npy', X_train_seq)
    np.save('npy/y_train_seq.npy', y_train_seq)
