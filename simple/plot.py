# PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


# # 特徴ベクトル
# x_train = np.load('newdata/newdata_npy/x_train.npy')
# y_train = np.load('newdata/newdata_npy/y_train.npy')
# train_x = np.load('newdata/newdata_npy/train_x.npy')
# X_train_i = np.load('newdata/newdata_npy/X_train_i.npy')
# X_train_bit= np.load('newdata/newdata_npy/X_train_bit.npy')
# train_x_kernel = np.load('newdata/newdata_npy/train_x_kernel.npy')

# # 実装段階の特徴ベクトル
# vH_i = np.load('newdata/newdata_npy/vH_i.npy')
# vH = np.load('newdata/newdata_npy/vH.npy')
# classify = np.load('newdata/newdata_npy/classify.npy')

# # 結合
# vH_i = np.concatenate([X_train_i, vH_i], 0)
# vH = np.concatenate([X_train, vH], 0)
# classify = np.concatenate([y_train, classify], 0)

# 前のデータ特徴
train_x1 = np.load('npy/train_x.npy')
y_train1 = np.load('npy/y_train.npy')
X_train_i1 = np.load('npy/new_train_x.npy')
X_train1 = np.load('npy/X_train_bit.npy')

#data = train_x
data = train_x1
print(data.shape)
# data_kernel = train_x_kernel
# print(data_kernel.shape)
#binary_data = X_train_i
binary_data = X_train_i1
print(binary_data.shape)
# bit_data = X_train
bit_data = X_train1
print(bit_data.shape)

# data = vH_i
# print(data.shape)
# binary_data = vH
# print(vH.shape)

# # サンプル
# np.random.seed(42)
# data = np.random.rand(10,540)
# print(data.shape)

#labels = y_train
labels = y_train1
print(labels.shape)

# labels = classify
# print(labels.shape)

# labels = np.random.randint(0,3,10)
# print(labels)

# PCAによる次元削減
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# pca1 = PCA(n_components=2)
# reduced_data1 = pca1.fit_transform(data_kernel)

pca2 = PCA(n_components=2)
reduced_data2 = pca2.fit_transform(binary_data)

pca3 = PCA(n_components=2)
reduced_data3 = pca3.fit_transform(bit_data)

#re = reduced_data2[720:,:]

# # t-sne
# tsne = TSNE(n_components=2, random_state=42)
# reduced_data = tsne.fit_transform(data)

# # tsne1 = TSNE(n_components=2, random_state=42)
# # reduced_data1 = tsne.fit_transform(data_kernel)

# tsne2 = TSNE(n_components=2, random_state=42)
# reduced_data2 = tsne2.fit_transform(binary_data)

# tsne3 = TSNE(n_components=2, random_state=42)
# reduced_data3 = tsne3.fit_transform(bit_data)

colors = ['r', 'g', 'b', 'purple']
#colors = ['r', 'g', 'b', 'purple', 'orange', 'y', 'c', 'm', 'brown']
#label_names = ['label0', 'label1', 'label2', 'label3']
label_names = [f'label{j}' for j in range(len(colors))]
# cmap = plt.get_cmap('tab10')
label_colors = [colors[label] for label in labels]

plt.figure(figsize=(10,7))
# ax = fig.add_subplot(111, projection='3d')
for i in range(len(colors)):
    indices = np.where(labels == i)
    if i in (0,1,2,3):
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], c=colors[i], label=label_names[i])
    else:
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], c=colors[i], label=label_names[i], alpha=0.1)
    #ax.scatter(reduced_data[indices, 0], reduced_data[indices,1], reduced_data[indices,2], c=colors[i], label=label_names[i])
    print(f'label{i}:{len(indices[0])}')
# plt.scatter(reduced_data[:,0],reduced_data[:,1], c=label_colors)
plt.title('pca')
#plt.title('t-sne')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
# ax.set_title('pca')
# ax.set_xlabel('component 1')
# ax.set_ylabel('component 2')
# ax.set_zlabel('component 3')
# ax.legend()
#plt.savefig('plot/pca.png')
plt.savefig('plot/pca_newdata.png')
plt.show()

plt.figure(figsize=(10,7))
#ax1 = fig.add_subplot(111, projection='3d')
for i in range(len(colors)):
    indices = np.where(labels == i)
    if i in (0,1,2,3):
        plt.scatter(reduced_data2[indices, 0], reduced_data2[indices, 1], c=colors[i], label=label_names[i])
    else:
        plt.scatter(reduced_data2[indices, 0], reduced_data2[indices, 1], c=colors[i], label=label_names[i], alpha=0.1)
    #ax1.scatter(reduced_data1[indices, 0], reduced_data1[indices,1], reduced_data1[indices,2], c=colors[i], label=label_names[i])
# plt.scatter(reduced_data1[:,0],reduced_data1[:,1], c=label_colors)
plt.title('pca')
#plt.title('t-sne')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
# ax1.set_title('pca')
# ax1.set_xlabel('component 1')
# ax1.set_ylabel('component 2')
# ax1.set_zlabel('component 3')
# ax1.legend()
#plt.savefig('plot/pca1.png')
plt.savefig('plot/pca_newdata1.png')
plt.show()

plt.figure(figsize=(10,7))
# # ax = fig.add_subplot(111, projection='3d')
for i in range(4):
    indices = np.where(labels == i)
    plt.scatter(reduced_data3[indices, 0], reduced_data3[indices, 1], c=colors[i], label=label_names[i])
#     #ax.scatter(reduced_data[indices, 0], reduced_data[indices,1], reduced_data[indices,2], c=colors[i], label=label_names[i])
#     #print(f'label{i}:{len(indices[0])}')
# # plt.scatter(reduced_data[:,0],reduced_data[:,1], c=label_colors)
plt.title('pca')
#plt.title('t-sne')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
# # ax.set_title('pca')
# # ax.set_xlabel('component 1')
# # ax.set_ylabel('component 2')
# # ax.set_zlabel('component 3')
# # ax.legend()
# #plt.savefig('plot/pca2.png')
plt.savefig('plot/pca_newdata2.png')
plt.show()

# # t=sne
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# np.random.seed(42)
# data = np.random.rand(10,540)

# tsne = TSNE(n_components=2, random_state=42)
# resuced_data = tsne.fit_transform(data)

# plt.scatter(reduced_data[:,0],reduced_data[:,1])
# plt.title('pca')
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.show

