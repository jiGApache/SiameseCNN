import random
import scipy.io
import codecs
import pandas as pd
import numpy as np

import torch
import torch.nn as nn


# chinese_dtst_reference = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')
# # print(chinese_dtst_reference)
# mat = scipy.io.loadmat('ChineseDataset\TrainingSet1\A0001.mat')
# print(mat)

# import matplotlib.pyplot as plt
# from PreprocessingFilters import filter1
# import math

# struct1 = np.ones((mat['ECG'][0][0][2].shape[0], 6)) / 5
# struct2 = np.ones((mat['ECG'][0][0][2].shape[0], 45)) / 5
# data1 = filter1(mat['ECG'][0][0][2], struct1, struct2)[:, 100:-100]

# mean = []
# if (15000 - data1.shape[1]) > 0:
#     for i in range(12):
#         mean.append(np.full(15000 - data1.shape[1], np.mean(data1[i])))
#     print(data1.shape)
#     ekg = np.column_stack([data1, mean])
# else:
#     ekg = data1[:, :15000]
# print(ekg.shape)

# _, axs = plt.subplots(2)
# axs[0].plot(ekg[1, :])

# means = [5.6671288368373204e-08, -5.672094472486019e-08, -1.1381123812568519e-07,
#                  -7.73628575187182e-10, 8.544064961353723e-08, -8.466800578420468e-08,
#                  -5.644898281745803e-08, -3.3201897366838757e-07, -6.639807663731727e-08,
#                  -1.9771499946997733e-08, -3.3253429074075554e-08, 1.487236435452322e-07]
# stds = [0.2305271687030844, 0.24780370485706876, 0.23155043161905942,
#         0.22074153961314927, 0.20708526280758174, 0.22153766144293813,
#         0.353942949952694, 0.3942397032518631, 0.4228515959530688,
#         0.436324876078121, 0.47316252072611537, 0.5328047065188085]

# for i in range(12):
#     ekg[i, :] = (ekg[i, :] - means[i]) / stds[i]
# axs[1].plot(ekg[1, :])

# plt.show()



# import math
# stats = {}
# dataframe = chinese_dtst_reference.loc[chinese_dtst_reference['Recording'] <= 'A2000'].reset_index(drop=True)
# max_shape = 0
# min_shape = math.inf
# for i in range(len(dataframe.index)):
#     current_mat = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + dataframe['Recording'][i] + '.mat')['ECG'][0][0][2]
#     current_shape = current_mat.shape[1]

#     if current_shape not in stats.keys(): stats[current_shape] = 1
#     else: stats[current_shape] += 1

#     if max_shape < current_shape: max_shape = current_shape
#     if min_shape > current_shape: min_shape = current_shape
# print(min_shape, max_shape)

# import matplotlib.pyplot as plt
# plt.pie(stats.values(), labels=stats.keys())
# plt.show()



# import math

# df = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')
# df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)

# channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

# for i in range(2000):
#     mat = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2]
#     for j in range(12):
#         channels_of_12[j].append(mat[j])

# means = []
# stds = []
# for channel in channels_of_12:
#     counter = 0

#     regular_sum = 0
#     squared_sum = 0

#     for element in channel:
#         counter += len(element)
#         regular_sum += sum(element)

#     for element in channel:
#         squared_sum += sum(pow(element - regular_sum / counter, 2))

#     means.append(regular_sum / counter)
#     stds.append(math.sqrt(squared_sum / counter))

# print('means: ', means)
# print('stds: ', stds)




# from pairsDataset import PairsDataset
# from torch.utils.data import DataLoader 
# import matplotlib.pyplot as plt
# import math
# import os

# ds = PairsDataset(same_size=True)
# dl = DataLoader(ds, batch_size=1, shuffle=True)

# for TS_T, label in dl:
#     TS1, TS2, label = TS_T[0], TS_T[1], label

#     plt.plot(TS1[0][1], 'r')
#     plt.plot(TS2[0][1], 'b')
#     print(TS1[0][1].std())

#     plt.show()
    
print( torch.cuda.is_available())