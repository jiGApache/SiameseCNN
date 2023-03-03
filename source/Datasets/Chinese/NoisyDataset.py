import os
from torch.utils.data import Dataset
from Datasets.Chinese.NoisyDatasetPreprocessing import prepare_dataset
import scipy
import math
import torch
import random
import numpy as np
import pandas as pd

class NoisyPairsDataset(Dataset):
    
    def __init__(self, labels = [1, 2], WITH_ROLL=False):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.path_append = ''
        if WITH_ROLL: self.path_append = '_Rolled'

        if (not os.path.exists(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')):
            os.mkdir(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')
        if (len(os.listdir(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}')) == 0):
            prepare_dataset(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\\')

        self.labels = labels

        df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
        df = df.loc[df['Recording'] <= 'A2000']

        self.dfs = []
        for label in self.labels:
            self.dfs.append(df.loc[
                (df['First_label'] == label) | \
                (df['Second_label'] == label) | \
                (df['Third_label'] == label)
            ].reset_index(drop=True))

        self.ds_len = 0
        for df in self.dfs:
            self.ds_len += len(df) * 2
        
    def __getitem__(self, index):

        # Getting pairs for each label - same labels in pair
        for df in self.dfs:

            if index < len(df):
                
                f_index = index
                s_index = (index + 1) % len(df)

                ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{df["Recording"][f_index]}.mat')['ECG']
                ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{df["Recording"][s_index]}.mat')['ECG']
                label = 1.

                return (
                        torch.as_tensor(ecg1, dtype=torch.float32),
                        torch.as_tensor(ecg2, dtype=torch.float32),
                    ), torch.as_tensor((label), dtype=torch.float32)
            
            else: 
                index -= len(df)
                continue

        
        # Getting pairs for each label - different labels in pair
        for df in self.dfs:

            if index < len(df):
            
                f_index = index
                df_index = np.random.randint(0, len(self.labels))
                while df.equals(self.dfs[df_index]):
                    df_index = np.random.randint(0, len(self.labels))
                s_index = np.random.randint(0, len(self.dfs[df_index]))
                
                ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{df["Recording"][f_index]}.mat')['ECG']
                ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.dfs[df_index]["Recording"][s_index]}.mat')['ECG']
                label = 0.

                return (
                        torch.as_tensor(ecg1, dtype=torch.float32),
                        torch.as_tensor(ecg2, dtype=torch.float32),
                    ), torch.as_tensor((label), dtype=torch.float32)
        
            else: 
                index -= len(df)
                continue

    def __len__(self):
        return  self.ds_len