from torch.utils.data import Dataset
import os
import scipy.io
import pandas as pd
import numpy as np
import torch
import random

class PairsDataset(Dataset):
    
    def __init__(self):

        data = scipy.io.arff.loadarff('ECG5000\ECG5000.arff')
        self.df = pd.DataFrame(data[0])

        self.df_with = []
        self.df_without = []

        self.df_with.append(self.df.loc[self.df['target'] == b'1'].reset_index(drop=True))
        self.df_with.append(self.df.loc[self.df['target'] == b'2'].reset_index(drop=True))
        self.df_with.append(self.df.loc[self.df['target'] == b'3'].reset_index(drop=True))
        self.df_with.append(self.df.loc[self.df['target'] == b'4'].reset_index(drop=True))
        self.df_with.append(self.df.loc[self.df['target'] == b'5'].reset_index(drop=True))

        self.df_without.append(self.df.loc[self.df['target'] != b'1'].reset_index(drop=True))
        self.df_without.append(self.df.loc[self.df['target'] != b'2'].reset_index(drop=True))
        self.df_without.append(self.df.loc[self.df['target'] != b'3'].reset_index(drop=True))
        self.df_without.append(self.df.loc[self.df['target'] != b'4'].reset_index(drop=True))
        self.df_without.append(self.df.loc[self.df['target'] != b'5'].reset_index(drop=True))

    def getEqPair(self, df, index):

        if (index >= 0) and (index < (df.shape[0])):
            counter = f_el_pos = 0
            s_el_pos = 1
            while (counter != index):
                f_el_pos += 1
                s_el_pos = (s_el_pos + 1) % df.shape[0]
                counter += 1

            ecg1 = np.expand_dims(np.array(df.iloc[f_el_pos, :-1], dtype=np.float32), 0)
            ecg2 = np.expand_dims(np.array(df.iloc[s_el_pos, :-1], dtype=np.float32), 0)

            return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32)
            ), torch.as_tensor((1.), dtype=torch.float32)

        else: index -= df.shape[0]

        return index

    def getDifPair(self, df1, df2, index):
        if (index >= 0) and (index < df1.shape[0]):
            rand_item = random.randint(0, (df2.shape[0]) - 1)

            random_ecg = np.expand_dims(np.array(df2.iloc[rand_item, :-1], dtype=np.float32), 0)
            ecg = np.expand_dims(np.array(df1.iloc[index, :-1], dtype=np.float32), 0)

            return (
                torch.as_tensor(ecg, dtype=torch.float32),#FloatTensor(ekg),
                torch.as_tensor(random_ecg, dtype=torch.float32),#FloatTensor(random_ekg)
            ), torch.as_tensor((0.), dtype=torch.float32)
        else:  index -= df1.shape[0]

        return index

    def __getitem__(self, index):
        
        for df in self.df_with:
            result = self.getEqPair(df, index)
            if type(result) == tuple:
                return result
            else: index = result

        for (with_df, without_df) in zip(self.df_with, self.df_without):
            result = self.getDifPair(with_df, without_df, index)
            if type(result) == tuple:
                return result
            else: index = result

        raise IndexError('Dataset index is out of range')

        return 

    def __len__(self):
        total_pairs = 0

        for df in self.df_with:
            total_pairs += df.shape[0]

        return total_pairs * 2