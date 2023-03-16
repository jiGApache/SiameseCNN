import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

import torch
import matplotlib.pyplot as plt
# from Datasets.Physionet.FewShotDataset import FewShotDataset
from Datasets.Chinese.FewShotDataset import FewShotDataset
from Models.SiameseModel import Siamese
from Models.EmbeddingModel import EmbeddingModule
from sklearn.neighbors import KNeighborsClassifier

def train(model, diagnoses, ECGs, n_neigh=3):    

    embeddings = []
    with torch.no_grad():
        for i in range (len(diagnoses)):
            embeddings.append(torch.squeeze(model(ECGs[i])).detach().numpy())

    X = embeddings
    y = diagnoses

    classifier = KNeighborsClassifier(n_neighbors=n_neigh)
    classifier.fit(X, y)

    return classifier

def test(model, classifier, diagnoses, ECGs):

    embeddings = []
    with torch.no_grad():
        for i in range (len(ECGs)):
            embeddings.append(torch.squeeze(model(ECGs[i])).detach().numpy())

    counter = 0
    for i in range(len(embeddings)):
        predicted_class = classifier.predict(embeddings[i].reshape(1, -1))
        if predicted_class == diagnoses[i]: counter += 1

    print('accuracy: ', counter / len(embeddings))


SHOT = 3
dataset = FewShotDataset(shot=SHOT)

if __name__ == '__main__':
    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()

    # print(train_diagnoses[0], train_diagnoses[5])
    # plt.plot(train_ECGs[5][0])
    # plt.show()
    # exit()

    # in1 = train_ECGs[2]
    # in2 = train_ECGs[3]
    # out_emb_1, out_emb_2 = model(in1, in2)
    # print(torch.cdist(out_emb_1, out_emb_2, p=2))
    # exit()

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(torch.load('nets\SCNN_d=1.0_with_tan_5_labels.pth'))
    embedding_model.train(False)

    # test_ecg = train_ECGs[0]
    # print(model(test_ecg, test_ecg)[0])
    # print(embedding_model(test_ecg))
    # exit()

    classifier = train(embedding_model, train_diagnoses, train_ECGs)
    test(embedding_model, classifier, test_diagnoses, test_ECGs)