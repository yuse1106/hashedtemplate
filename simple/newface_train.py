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
from skimage.exposure import rescale_intensity
from utils import *

#テンプレートマッチング
def template(eigenVectors):
    eigenFaces = []
    # 固有値画像に変換
    for i, eigenVector in enumerate(eigenVectors):
        eigenFace = eigenVector.reshape(size)  
        # 画像を保存するため
        eigenFace1 = rescale_intensity(eigenFace, out_range=(0,255))   
        eigenFace1 = np.dstack([eigenFace1.astype("uint8")])
        # cv2.imshow("com", eigenFace1)
        # cv2.waitKey(0)
        # cv2.imread()
        os.makedirs('new_eigen/', exist_ok=True)
        cv2.imwrite(f'new_eigen/eigen_{i}.jpg', eigenFace1)
        eigenFaces.append(eigenFace)
    return eigenFaces



if __name__ == '__main__':
    start_time = time.time()

    eigenVectors = np.load('npy/eigenVectors.npy')
    eigenFaces = np.load('npy/eigenFaces.npy')
    # eigenFaces = np.load('npy/eigen_back.npy')
    x_train = np.load('npy/x_train.npy')
    y_train = np.load('npy/y_train.npy')

    # 訓練データにデータを加える
    path = "../add_dataset/add_train"
    x_train, y_train = add_data(path, x_train, y_train)
    print(x_train.shape)
    
    # 画像サイズ
    size = x_train[0].shape
    print(size)
    
    #事前学習データ，固有値画像との相関
    train_x = np.ones((len(x_train), len(eigenFaces)))
    for i,img in enumerate(x_train):
        for j,eigenFace in enumerate(eigenFaces):
            result = cv2.matchTemplate(img, eigenFace, cv2.TM_CCORR)
            train_x[i,j]=result

    # np.save('npy/train_x.npy', train_x)

    # 状態変換行列（ハッシュ関数生成）
    num_bit = 4
    Fmap = {'nu': 1e-5, 'lambda': 1e-2}
    trans = hadamard2(train_x, y_train, num_bit, Fmap)
    trans = trans.astype(np.float32)
    
    new_trans = np.transpose(trans)
    # ハッシュドテンプレート
    new_eigenVectors = np.dot(new_trans, eigenVectors)
    # ハッシュドテンプレート生成
    new_eigenFaces = template(new_eigenVectors)

    #学習データ，ハッシュドテンプレートとの相関
    new_train_x = np.ones((len(x_train), len(new_eigenFaces)))
    for i,img in enumerate(x_train):
        for j,new_eigenFace in enumerate(new_eigenFaces):
            result = cv2.matchTemplate(img, new_eigenFace, cv2.TM_CCORR)
            new_train_x[i,j]=result

    # np.save('npy/new_train_x.npy', new_train_x)

    # #カーネル
    # Ntrain = train_x.shape[0]
    # n_anchor = Ntrain
    # index = np.random.choice(Ntrain, n_anchor, replace=False)
    # X_anchor = train_x[index,:]
    # #ガウスカーネル
    # Vsigma = 0.3
    # train_X = np.exp(-sqdist(train_x, X_anchor) / (2*Vsigma*Vsigma))

    # 0以上を1，それ以外を-1にする
    # X_train = X_train_i >0
    # X_train = X_train.astype(int)
    # signによる識別関数
    X_train = np.sign(new_train_x)
    X_train = X_train.astype(int)

    np.save('npy/X_train_bit.npy', X_train)

    # バイナリビットに変換しないときの最初の特徴ベクトルを取得
    X_train_seq = []
    # 変換したとき重複ないように
    X_train_seq, indice = np.unique(X_train, axis = 0, return_index = True)
    X_train_seq = np.array(X_train_seq)
    print(X_train_seq)
    y_train_seq = []
    y = y_train[indice]
    y_train_seq.append(y)
    print(y_train_seq)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")

    os.makedirs('new_npy', exist_ok=True)
    np.save('new_npy/new_eigen1.npy', new_eigenFaces)
    np.save('new_npy/X_train_seq.npy', X_train_seq)
    np.save('new_npy/y_train_seq.npy', y_train_seq)