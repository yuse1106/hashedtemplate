# テンプレート画像を読み込み，固有値テンプレートを作成する
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import os
import sys
import cv2
import glob
import numpy as np
import math
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import time
from utils import *

#テンプレートマッチング
def template(mean, eigenVectors):
    # Create a container to hold eigen faces.
    eigentemps = []
    # Reshape eigen vectors to eigen faces.
    for i, eigenVector in enumerate(eigenVectors):
        eigentemp = eigenVector.reshape(size) 
        eigentemp1 = rescale_intensity(eigentemp, out_range=(0,255))   
        eigentemp1 = np.dstack([eigentemp1.astype("uint8")])   
        os.makedirs('eigenimage', exist_ok=True) 
        cv2.imwrite(f'eigenimage/eigen_{i}.jpg', eigentemp1)
        # cv2.imwrite(f'eigenimage/eigenimage/eigen_{i}.jpg', eigenFace1)
        eigentemps.append(eigentemp)
    return eigentemps


if __name__ == '__main__':
    start_time = time.time()
    #PCA後のテンプレート数，少なすぎると情報がなくなってしまう
    NUM_EIGEN_FACES = 450
    #画像ファイルディレクトリ
    path = "../tra"
    # x_train,y_train,apple_rotate,lemon_rotate,orange_rotate = get_data(path)
    x_train, y_train, label = get_data(path, 0)

    os.makedirs('npy', exist_ok=True)
    np.save('npy/x_train.npy', x_train)
    np.save('npy/y_train.npy', y_train)

    #x = x_train[0]
    size = x_train[0].shape
    data = createMatrix(x_train[0:1080], size)
    # Compute the eigenvectors from the stack of images created.
    print("Calculating PCA", end = "...")
    mean, eigenVectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN_FACES)
    np.save('npy/eigenVectors.npy', eigenVectors)

    # eigentemps = []
    # for i in range(label):
    #     data = createMatrix(x_train[i*360:(i+1)*360], size)
    #     mean, eigenVectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN_FACES)
    #     eigen = template(mean, eigenVectors)
    #     eigentemps.append(eigen)
    # eigentemps = np.concatenate(eigentemps)
    # eigentemps = np.array(eigentemps)

    eigentemps = template(mean, eigenVectors)
    eigentemps = np.array(eigentemps)
    print("eigenFaceの大きさ", eigentemps.shape)
    print(len(x_train))
    np.save('npy/eigenFaces.npy', eigentemps)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"実行時間:{minutes}分{seconds}秒")
    