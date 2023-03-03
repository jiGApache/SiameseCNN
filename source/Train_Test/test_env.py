import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

import random
import scipy.io
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Models.SiameseModel import Siamese
from Models.EmbeddingModel import EmbeddingModule
# from Datasets.Physionet.NoisyDataset import PairsDataset
from Datasets.Chinese.NoisyDataset import NoisyPairsDataset

ds_noisy = NoisyPairsDataset([1, 2, 5])
# pair, label = ds_noisy.__getitem__(700) # Different but looks same: 8, 10

# model = Siamese()
# model.load_state_dict(torch.load('nets\SCNN.pth'))

# in1 = pair[0][None, :, :]
# in2 = pair[1][None, :, :]

# model.train(False)
# print('predicted: ', model(in1, in2).item())
# print('true: ', label.item())

# fig, axs = plt.subplots(2)
# axs[0].plot(pair[0][0])
# axs[1].plot(pair[1][0])
# plt.ylim(-2, 2)
# plt.show()

################################################################################

# # ecg = scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\A0101.mat')['ECG'][0][0][2]
# ecg = scipy.io.loadmat('Data\ChineseDataset\FilteredECG\A0101.mat')['ECG']
# # ecg = scipy.io.loadmat('Data\ChineseDataset\PreparedDataset_Noisy\A0101.mat')['ECG']
# plt.plot(ecg[2])
# plt.ylim(-2, 2)
# # plt.ylim(-8, 8)
# plt.show()

################################################################################

model = Siamese()
model.load_state_dict(torch.load('nets\SCNN.pth'))
model.train(False)

embedding_model = EmbeddingModule()
embedding_model.load_state_dict(model.state_dict())
embedding_model.train(False)


ecgs_1_files = ['A0149', 'A0157', 'A0164', 'A0166', 'A0170', 'A0173', 'A0175', 'A0176', 'A0177', 'A0179']
ecgs1 = []
for name in ecgs_1_files:
    ecgs1.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{name}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])

ecgs_2_files = ['A0184', 'A0186', 'A0198', 'A0203', 'A0205', 'A0214', 'A0217', 'A0220', 'A0222', 'A0231']
ecgs2 = []
for name in ecgs_2_files:
    ecgs2.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{name}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])

ecgs_3_files = ['A0188', 'A0212', 'A0223', 'A0236', 'A0238', 'A0239', 'A0240', 'A0243', 'A0246', 'A0248']
ecgs3 = []
for name in ecgs_3_files:
    ecgs3.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{name}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])

ecgs_8_files =['A0154', 'A0165', 'A0185', 'A0187', 'A0194', 'A0195', 'A0196', 'A0201', 'A0232', 'A0234']
ecgs8 = []
for name in ecgs_8_files:
    ecgs8.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{name}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])

embeddings1 = []
embeddings2 = []
embeddings3 = []
embeddings8 = []

for ecg in ecgs1:    embeddings1.append(torch.squeeze(embedding_model(ecg)).detach().numpy())
for ecg in ecgs2:    embeddings2.append(torch.squeeze(embedding_model(ecg)).detach().numpy())
for ecg in ecgs3:    embeddings3.append(torch.squeeze(embedding_model(ecg)).detach().numpy())
for ecg in ecgs8:    embeddings8.append(torch.squeeze(embedding_model(ecg)).detach().numpy())

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(embeddings1+embeddings2+embeddings3+embeddings8)
# embeddings1 = pca.transform(embeddings1)
# embeddings2 = pca.transform(embeddings2)
# embeddings3 = pca.transform(embeddings3)
# embeddings8 = pca.transform(embeddings8)

# plt.scatter(x=embeddings1[:, 0], y=embeddings1[:, 1], c='green')
# plt.scatter(x=embeddings2[:, 0], y=embeddings2[:, 1], c='red')
# # plt.scatter(x=embeddings3[:, 0], y=embeddings3[:, 1], c='purple')
# # plt.scatter(x=embeddings8[:, 0], y=embeddings8[:, 1], c='blue')
# plt.show()

import umap
fit = umap.UMAP()
fit.fit(embeddings1+embeddings2+embeddings3+embeddings8)
embeddings1 = fit.transform(embeddings1)
embeddings2 = fit.transform(embeddings2)
embeddings3 = fit.transform(embeddings3)
embeddings8 = fit.transform(embeddings8)

plt.scatter(x=embeddings1[:, 0], y=embeddings1[:, 1], c='green')
plt.scatter(x=embeddings2[:, 0], y=embeddings2[:, 1], c='red')
# plt.scatter(x=embeddings3[:, 0], y=embeddings3[:, 1], c='purple')
# plt.scatter(x=embeddings8[:, 0], y=embeddings8[:, 1], c='blue')
plt.show()