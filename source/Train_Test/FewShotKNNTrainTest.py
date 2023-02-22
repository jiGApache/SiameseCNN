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
            ecg_input = ECGs[i][None, :, :]
            embeddings.append(torch.squeeze(model(ecg_input)).detach().numpy())

    X = embeddings
    y = diagnoses

    classifier = KNeighborsClassifier(n_neighbors=n_neigh)
    classifier.fit(X, y)

    return classifier

def test(model, classifier, diagnoses, ECGs):

    embeddings = []
    with torch.no_grad():
        for i in range (len(ECGs)):
            ecg_input = ECGs[i][None, :, :]
            embeddings.append(torch.squeeze(model(ecg_input)).detach().numpy())

    counter = 0
    for i in range(len(embeddings)):
        predicted_class = classifier.predict(embeddings[i].reshape(1, -1))
        if predicted_class == diagnoses[i]: counter += 1

    print('accuracy: ', counter / len(embeddings))


SHOT = 10
dataset = FewShotDataset(shot=SHOT)

if __name__ == '__main__':
    train_diagnoses, train_ECGs = dataset.get_train_data()
    test_diagnoses, test_ECGs = dataset.get_test_data()

    import pandas as pd
    import scipy
    df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv')
    df_with_class = df.loc[(df['First_label'] == 1) & (df['Recording'] <= 'A2000')]
    ECGs = []
    for i in range(15):
        ECGs.append(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{df_with_class.iloc[i]["Recording"]}.mat')['ECG'][:, :2900])


    # print(train_diagnoses[0], train_diagnoses[5])
    # plt.plot(train_ECGs[5][0])
    # plt.show()
    # exit()

    in1 = torch.as_tensor(ECGs[0][None, :, :], dtype=torch.float32)
    in2 = torch.as_tensor(ECGs[9][None, :, :], dtype=torch.float32)
    # in2 = torch.as_tensor(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\A0025.mat')['ECG'][None, :, :2900], dtype=torch.float32)

    model = Siamese()
    model.load_state_dict(torch.load('nets\SCNN.pth'))
    model.train(False)
    out_emb_1, out_emb_2 = model(in1, in2)
    print(torch.cdist(out_emb_1, out_emb_2, p=2))
    exit()

    embedding_model = EmbeddingModule()
    embedding_model.load_state_dict(model.state_dict())
    embedding_model.train(False)

    classifier = train(embedding_model, train_diagnoses, train_ECGs)
    test(embedding_model, classifier, test_diagnoses, test_ECGs)