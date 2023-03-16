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

# ds_noisy = NoisyPairsDataset()
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


train_labels = [1, 3, 5, 7, 9]
test_labels = []

# train_labels = [1, 2, 3, 4, 5, 6, 7]
# test_labels = []

# PCA = 1
# UMAP = 2
METHOD = 2


distances = [0.5, 1., 2., 3., 4., 5.]
for plot_index, distance in enumerate(distances):

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(torch.load(f'nets\SCNN_d={distance}_labels={len(train_labels)}.pth'))
    embedding_model.train(False)


    train_ECGs = []
    test_ECGs = []
    df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
    ELEMENTS_PER_CLASS = 10


    for label in train_labels:
        labeld_df = df.loc[((df['First_label'] == label) & (df['Second_label'] != label) & (df['Third_label'] != label))].reset_index(drop=True)
        for i in range(ELEMENTS_PER_CLASS):
            train_ECGs.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{labeld_df["Recording"][i]}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])
    
    for label in test_labels:
        labeld_df = df.loc[((df['First_label'] == label) & (df['Second_label'] != label) & (df['Third_label'] != label))].reset_index(drop=True)
        for i in range(ELEMENTS_PER_CLASS):
            test_ECGs.append(torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{labeld_df["Recording"][i]}.mat')['ECG'], dtype=torch.float32)[None, :, 100:3100])


    train_embeddings = []
    test_embeddings = []

    for ecg in train_ECGs:
        train_embeddings.append(torch.squeeze(embedding_model(ecg)).detach().numpy())

    for ecg in test_ECGs:
        test_embeddings.append(torch.squeeze(embedding_model(ecg)).detach().numpy())



    if METHOD == 1:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(train_embeddings + test_embeddings)
        if len(train_labels) > 0: tf_train_embeds = pca.transform(train_embeddings)
        if len(test_labels) > 0: tf_test_embeds = pca.transform(test_embeddings)
    else:
        import umap
        fit = umap.UMAP()
        fit.fit(train_embeddings + test_embeddings)
        if len(train_labels) > 0: tf_train_embeds = fit.transform(train_embeddings)
        if len(test_labels) > 0: tf_test_embeds = fit.transform(test_embeddings)


    plt.subplot(2, 3, plot_index+1)
    plt.subplot(2, 3, plot_index+1).set_xlabel(f'd={distance}')
    for i in range(len(train_labels)):
        plt.scatter(
            tf_train_embeds[ELEMENTS_PER_CLASS*i:ELEMENTS_PER_CLASS*(i+1), 0],
            tf_train_embeds[ELEMENTS_PER_CLASS*i:ELEMENTS_PER_CLASS*(i+1), 1],
            label=train_labels[i])
    for i in range(len(test_labels)):
        plt.scatter(
            tf_test_embeds[ELEMENTS_PER_CLASS*i:ELEMENTS_PER_CLASS*(i+1), 0],
            tf_test_embeds[ELEMENTS_PER_CLASS*i:ELEMENTS_PER_CLASS*(i+1), 1],
            label=test_labels[i])

    plt.margins(0.35, 0.35)
    plt.legend()
    
plt.show()


################################################################################


# import json
# distances = [0.5, 1., 2., 3., 4., 5.]
# step = 150
# for distance in distances:
#     with open(f'history\history_d={distance}_labels=5.json') as json_file:
#         data = json.load(json_file)
#         plt.plot(data['epochs'], data['test_losses'], label=str(distance))
#         plt.legend()
# plt.show()