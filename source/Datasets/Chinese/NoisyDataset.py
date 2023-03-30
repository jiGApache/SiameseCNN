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
    
    def __init__(self, labels = [3, 5], WITH_ROLL=False):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Preparing dataset #########################################################################
        self.path_append = '_Rolled' if WITH_ROLL == True else ''
        if (not os.path.exists(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}')):
            os.mkdir(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}')
        if (len(os.listdir(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}')) == 0):
            prepare_dataset(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\\')
        #############################################################################################

        self.NORMAL_LABEL = 1
        self.labels = labels

        df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
        df = df.loc[df['Recording'] <= 'A4470']

        self.normal_df = df.loc[
                (df['First_label'] == self.NORMAL_LABEL) | \
                (df['Second_label'] == self.NORMAL_LABEL) | \
                (df['Third_label'] == self.NORMAL_LABEL)
            ].reset_index(drop=True)

        self.dfs = []
        for label in self.labels:
            self.dfs.append(df.loc[
                (df['First_label'] == label) | \
                (df['Second_label'] == label) | \
                (df['Third_label'] == label)
            ].reset_index(drop=True))

        self.ds_len = len(self.normal_df) * 2 + len(self.normal_df) * len(self.dfs)
        
    def __getitem__(self, index):

        # Pairs with equal normal labels | step = 1
        if index < len(self.normal_df):

            f_index = index
            s_index = (index + 1) % len(self.normal_df)

            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][s_index]}.mat')['ECG']
            label = 1.

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32),
                ), torch.as_tensor((label), dtype=torch.float32)
            
        else: index -= len(self.normal_df)     

        # Pairs with equal normal labels | step = (amount of normal ECG) // 2
        if index < len(self.normal_df):

            f_index = index
            s_index = (index + len(self.normal_df) // 2) % len(self.normal_df)

            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][s_index]}.mat')['ECG']
            label = 1.

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32),
                ), torch.as_tensor((label), dtype=torch.float32)
            
        else: index -= len(self.normal_df)        
        


        
        # Getting pairs for normal label - different labels in pair

        f_index = index // len(self.dfs)

        df = self.dfs[index % len(self.dfs)]
        s_index = np.random.randint(0, len(df))

        ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')['ECG']
        ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\\1\PreparedDataset_Noisy{self.path_append}\{df["Recording"][s_index]}.mat')['ECG']
        label = 0.

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32),
            ), torch.as_tensor((label), dtype=torch.float32)

    def __len__(self):
        return  self.ds_len