import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

import torch
import matplotlib.pyplot as plt
from Datasets.FewShotDataset import FewShotDataset
from Filtering.PreprocessingFilters import filter1
from Models.SiameseModel import Siamese
from Models.EmbeddingModel import EmbeddingModule
from sklearn.neighbors import KNeighborsClassifier
from Datasets.NoisyDataset import NoisyPairsDataset as NS_dataset

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


SHOT = 10
dataset = FewShotDataset(shot=SHOT)

if __name__ == '__main__':
    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()

    # print(train_diagnoses[0], train_diagnoses[1])
    # plt.plot(train_ECGs[0][0])
    # plt.show()
    # exit()

    in1 = train_ECGs[0][None, :, :]
    in2 = train_ECGs[1][None, :, :]

    model = Siamese()
    model.load_state_dict(torch.load('nets\SCNN.pth'))
    model.train(False)
    print(model(in1, in2))
    exit()

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(model.state_dict())
    embedding_model.train(False)

    classifier = train(embedding_model, train_diagnoses, train_ECGs)
    test(embedding_model, classifier, test_diagnoses, test_ECGs)