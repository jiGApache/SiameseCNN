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

def get_channel_means_stds():
    dataset = NS_dataset()
    channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for i in range(0, len(dataset), 2):
        pair, _ = dataset.__getitem__(i)
        ecg_fragment = pair[0]
        for i in range(ecg_fragment.shape[0]):
            channels_of_12[i].append(ecg_fragment[i])

    means = []
    stds = []
    for channel in channels_of_12:
        
        counter = 0
        regular_sum = 0
        squared_sum = 0

        for element in channel:
            counter += len(element)
            regular_sum += sum(element)
        for element in channel:
            squared_sum += sum(pow(element - regular_sum / counter, 2))

        means.append(regular_sum / counter)
        stds.append(math.sqrt(squared_sum / (counter - 1)))

    return means, stds


def filter_ecg(ekg):
    struct1 = np.ones((ekg.shape[0], 6)) / 5
    struct2 = np.ones((ekg.shape[0], 45)) / 5
    data = filter1(ekg, struct1, struct2)[:, 50:-50]
    return data


def prepare_ECG(ECGs):
    # Filtering
    for i in range(len(ECGs)):
        ECGs[i] = filter_ecg(ECGs[i])

    # means, stds = get_channel_means_stds()
    means = [0.0016, 0.0003, 0.0006, 0.0009, 0.0017, 0.0005, -0.0020, -0.0008, 0.0005, 0.0009, -1.4044e-05, 0.0003]
    stds = [1.0365105173338283, 1.0212978097844168, 1.028844629063083, 1.0287473539964986, 1.0416054262597099, 1.0228705546648325, 1.021233960122595, 1.0075333082879048, 1.0084018610427552, 1.021604512664429, 1.0291869595234864, 1.0514313909410649]
    for i in range(len(ECGs)):
        new_ecg = []
        for k in range(12):
            if np.std(ECGs[i][k]) < 1e-8: std = 1
            else: std = np.std(ECGs[i][k])
            new_ecg.append((ECGs[i][k] - np.mean(ECGs[i][k])) / std)
        
        new_ecg = np.array(new_ecg)

        for k in range(12):
            new_ecg[k] = new_ecg[k] * stds[k] + means[k]

        ECGs[i] = torch.as_tensor(new_ecg, dtype=torch.float32)

    return ECGs


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

    print('[ ', classifier.classes_[0], classifier.classes_[1], classifier.classes_[2], classifier.classes_[3], classifier.classes_[4], ' ]')
    for i in range(len(embeddings)):

        if to_skip.__contains__(i): continue

        preds = classifier.predict_proba(embeddings[i])
        print('[', preds[0], preds[1], preds[2], preds[3], preds[4], ' ]')
        print('correct: ', diagnoses[i], '\n')


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
SHOT = 3

dataset = FewShotDataset(shot=SHOT)

# mins = [-98.8017, -102.2583, -128.8134, -78.5670, -113.7432, -125.9245, -59.9016, -65.0385, -60.0551, -55.6489, -56.8669, -59.4899]   
# maxs = [99.2798, 58.8441, 85.8087, 87.4079, 121.5268, 64.1094, 56.7045, 52.4901, 55.6870, 51.6214, 52.7730, 70.2337]

if __name__ == '__main__':
    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()

    train_ECGs = prepare_ECG(train_ECGs)
    test_ECGs = prepare_ECG(test_ECGs)

    model = Siamese()
    model.load_state_dict(torch.load('nets\SCNN.pth'))
    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(model.state_dict())
    embedding_model.train(False)

    classifier = train(embedding_model, train_diagnoses, train_ECGs)
    test(embedding_model, classifier, test_diagnoses, test_ECGs)
