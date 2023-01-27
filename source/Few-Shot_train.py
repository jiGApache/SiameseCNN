import pandas as pd
import random
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from model import Siamese
from EmbeddingModel import EmbeddingModule

# ECG_META = {}

# def get_labels(header):
#     labels = list()
#     for l in header.split('\n'):
#         if l.startswith('#Dx'):
#             try:
#                 entries = l.split(': ')[1].split(',')
#                 for entry in entries:
#                     labels.append(entry.strip())
#             except:
#                 pass
#     return labels

# def pad_dict_list(dict_list, padel):
#     lmax = 0
#     for lname in dict_list.keys():
#         lmax = max(lmax, len(dict_list[lname]))
#     for lname in dict_list.keys():
#         ll = len(dict_list[lname])
#         if  ll < lmax:
#             dict_list[lname] += [padel] * (lmax - ll)
#     return dict_list



# for i in range(1, 10345):
#     file = open(f'GeorgiaDataset\E{str(i).zfill(5)}.hea', 'r')
#     loc_labels = get_labels(file.read())
#     for label in loc_labels:
#         if label not in ECG_META.keys():
#             ECG_META[label] = []
#         ECG_META[label].append(f'E{str(i).zfill(5)}.mat')
#     file.close()    

# ECG_META = pad_dict_list(ECG_META, '')

# df = pd.DataFrame.from_dict(ECG_META).transpose()
# pd.DataFrame.to_csv(df, 'GeorgiaDataset\META_DATA.csv', index_label='diagnosis')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

df1 = pd.read_csv('GeorgiaDataset\META_DATA.csv', index_col='diagnosis')
# print(df1)

diagnoses = []
for i in range(5): 
    index = random.randint(0, len(df1.index)-1)
    diagnoses.append(df1.index[index])
# print(diagnoses)

ECGs = []
for diag in diagnoses:
    diag_ecg = []
    for i in range(3):
        diag_ecg.append(scipy.io.loadmat(f'GeorgiaDataset\{df1.loc[diag][i]}')['val'][:, :2900])
    ECGs.append(diag_ecg)
# print(ECGs)

mins = [-98.8017, -102.2583, -128.8134, -78.5670, -113.7432, -125.9245, -59.9016, -65.0385, -60.0551, -55.6489, -56.8669, -59.4899]   
maxs = [99.2798, 58.8441, 85.8087, 87.4079, 121.5268, 64.1094, 56.7045, 52.4901, 55.6870, 51.6214, 52.7730, 70.2337]
for i in range(len(diagnoses)):
    for j in range(3):
        new_ecg = []
        for k in range(12):
            new_ecg.append((ECGs[i][j][k] - mins[k]) / (maxs[k] - mins[k]))
        ECGs[i][j] = torch.as_tensor(new_ecg, dtype=torch.float32)
# plt.plot(ECGs[0][0][0])
# plt.show()

model = Siamese()
model.load_state_dict(torch.load('nets\SCNN.pth'))
embedding_model = EmbeddingModule()
embedding_model.load_state_dict(model.state_dict())
embedding_model.train(False)

embeddings = []
with torch.no_grad():
    for i in range (len(diagnoses)):
        diag_embeddings = []
        for j in range(3):
            ecg_input = ECGs[i][j][None, :, :]
            embeddings.append(torch.squeeze(embedding_model(ecg_input)).detach().numpy())

# RESULT = {}
# for i, diag in enumerate(diagnoses):
#     if diag not in RESULT.keys():
#         RESULT[diag] = []
#     for j in range(3):
#         RESULT[diag].append({f'{df1.loc[diag][j]}': embeddings[i][j]})
# print(RESULT)

from sklearn.neighbors import KNeighborsClassifier
X = embeddings
y = []
for el in diagnoses:
    for i in range(3):
        y.append(el)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)


test_diagnose = diagnoses[0]
print('Testing on diagnose: ', test_diagnose, ', file: ', df1.loc[test_diagnose][8])
ecg = scipy.io.loadmat(f'GeorgiaDataset\{df1.loc[test_diagnose][7]}')['val'][:, :2900]
new_ecg = []
for i in range(12):
    new_ecg.append((ecg[i] - mins[i]) / (maxs[i] - mins[i]))
ecg = torch.as_tensor(new_ecg, dtype=torch.float32)[None, :, :]
with torch.no_grad():
    test_embedding = torch.squeeze(embedding_model(ecg)).detach().numpy()
print(classifier.classes_)
print(classifier.predict_proba([test_embedding]))