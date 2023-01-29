import os
import pandas as pd
import random
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from FewShotDataset import FewShotDataset
from PreprocessingFilters import filter1
from model import Siamese
from EmbeddingModel import EmbeddingModule
from sklearn.neighbors import KNeighborsClassifier
from NoisyDataset import NoisyPairsDataset as NS_dataset
import math

def train(model, diagnoses, ECGs, n_neigh=3):    

    embeddings = []
    with torch.no_grad():
        for i in range (len(diagnoses)):
            ecg_input = ECGs[i][None, :, :]
            embeddings.append(torch.squeeze(model(ecg_input)).detach().numpy())

    X = embeddings
    y = diagnoses

    classifier = KNeighborsClassifier(n_neighbors=n_neigh)
    classifier.fit(X, y)

    return classifier

def test(model, classifier, diagnoses, ECGs):

    embeddings = []
    to_skip = []
    with torch.no_grad():
        for i in range (len(ECGs)):

            if ECGs[i].shape[1] < 2900: 
                to_skip.append(i)
                continue

            ecg_input = ECGs[i][None, :, :]
            embeddings.append(torch.squeeze(model(ecg_input)).detach().numpy())

    counter = 0
    for i in range(len(embeddings)):

        if to_skip.__contains__(i): continue

        predicted_class = classifier.predict(embeddings[i].reshape(1, -1))
        if predicted_class == diagnoses[i]: counter += 1
        # preds = classifier.predict_proba(embeddings[i])
        # print('[', preds[0], preds[1], preds[2], preds[3], preds[4], ' ]')
        # print('correct: ', diagnoses[i], '\n')

    print('accuracy: ', counter / len(embeddings))


SHOT = 5

dataset = FewShotDataset(shot=SHOT)

# mins = [-98.8017, -102.2583, -128.8134, -78.5670, -113.7432, -125.9245, -59.9016, -65.0385, -60.0551, -55.6489, -56.8669, -59.4899]   
# maxs = [99.2798, 58.8441, 85.8087, 87.4079, 121.5268, 64.1094, 56.7045, 52.4901, 55.6870, 51.6214, 52.7730, 70.2337]

if __name__ == '__main__':
    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()

    # print(train_diagnoses[4], train_diagnoses[5])
    # plt.plot(train_ECGs[5][0])
    # plt.show()
    # exit()

    in1 = test_ECGs[0][None, :, :]
    in2 = test_ECGs[1][None, :, :]

    model = Siamese()
    model.load_state_dict(torch.load('nets\SCNN.pth'))
    print(model(in1, in2))
    exit()

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(model.state_dict())
    embedding_model.train(False)

    classifier = train(embedding_model, train_diagnoses, train_ECGs)
    test(embedding_model, classifier, test_diagnoses, test_ECGs)