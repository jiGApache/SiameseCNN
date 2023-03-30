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

# ds_noisy = NoisyPairsDataset(labels=[8, 9])
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

# # ecg = scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\A0011.mat')['ECG'][0][0][2][:, :5000]
# ecg = scipy.io.loadmat('Data\ChineseDataset\FilteredECG\A0011.mat')['ECG'][:, :5000]
# # ecg = scipy.io.loadmat('Data\ChineseDataset\PreparedDataset_Noisy\A0011.mat')['ECG']
# plt.plot(ecg[0])
# plt.ylim(-1, 1)
# # plt.ylim(-8, 8)
# plt.show()

################################################################################


normal_label = 1
abnormal_labels = [8, 9]
# abnormal_labels = [2, 4, 6, 8]

# PCA = 1
# UMAP = 2
METHOD = 2


distances = [1., 2., 3., 4., 8., 20.]
for plot_index, distance in enumerate(distances):

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(torch.load(f'nets\SCNN_d={distance}_labels=2.pth'))
    embedding_model.train(False)


    normal_ECGs = []
    abnormal_ECGs = []
    df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
    # df = df.loc[df['Recording'] <= 'A4470'].reset_index(drop=True)
    df = df.loc[df['Recording'] >= 'A4471'].reset_index(drop=True)
    ELEMENTS_PER_CLASS = 50

    normal_df = df.loc[(df['First_label'] == normal_label)].reset_index(drop=True)
    for i in range(ELEMENTS_PER_CLASS):
        normal_ECGs.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\\3\PreparedDataset_Noisy\{normal_df["Recording"][i]}.mat')['ECG'], dtype=torch.float32)[None, :, :])

    for label in abnormal_labels:
        labeld_df = df.loc[((df['First_label'] == label) | (df['Second_label'] == label) | (df['Third_label'] == label))].reset_index(drop=True)
        for i in range(ELEMENTS_PER_CLASS):
            abnormal_ECGs.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\\3\PreparedDataset_Noisy\{labeld_df["Recording"][i]}.mat')['ECG'], dtype=torch.float32)[None, :, :])


    normal_embeddings = []
    abnormal_embeddings = []

    for ecg in normal_ECGs:
        normal_embeddings.append(torch.squeeze(embedding_model(ecg)).detach().numpy())

    for ecg in abnormal_ECGs:
        abnormal_embeddings.append(torch.squeeze(embedding_model(ecg)).detach().numpy())



    if METHOD == 1:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(abnormal_embeddings + normal_embeddings)
        tf_normal_embeds = pca.transform(normal_embeddings)
        if len(abnormal_labels) > 0: tf_abnormal_embeds = pca.transform(abnormal_embeddings)
    else:
        import umap
        fit = umap.UMAP()
        fit.fit(abnormal_embeddings + normal_embeddings)
        tf_normal_embeds = fit.transform(normal_embeddings)
        if len(abnormal_labels) > 0: tf_abnormal_embeds = fit.transform(abnormal_embeddings)


    plt.subplot(2, 3, plot_index+1)
    plt.subplot(2, 3, plot_index+1).set_xlabel(f'd={distance}')
    plt.scatter(
        tf_normal_embeds[:, 0],
        tf_normal_embeds[:, 1],
        label='normal')
    plt.scatter(
        tf_abnormal_embeds[:, 0],
        tf_abnormal_embeds[:, 1],
        label='abnormal')

    plt.margins(0.35, 0.35)
    plt.legend()
    
plt.show()


################################################################################


# import json
# distances = [1., 2., 3., 4., 8., 20.]
# for distance in distances:
#     with open(f'history\history_d={distance}_labels=2.json') as json_file:
#         data = json.load(json_file)
#         plt.plot(data['epochs'], data['test_losses'], label=str(distance))
#         plt.legend()
# plt.show()