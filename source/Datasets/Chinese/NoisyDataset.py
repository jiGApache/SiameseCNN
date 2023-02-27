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
    
    def __init__(self, WITH_ROLL=False):

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

        self.normal_diagnose = 1
        self.abnormal_diagnose = 2

        df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
        df = df.loc[df['Recording'] <= 'A2000']

        self.normal_df = df.loc[(df['First_label'] == self.normal_diagnose) | \
                                (df['Second_label'] == self.normal_diagnose) | \
                                (df['Third_label'] == self.normal_diagnose)].reset_index(drop=True)
        
        self.abnormal_df = df.loc[(df['First_label'] == self.abnormal_diagnose) | \
                                  (df['Second_label'] == self.abnormal_diagnose) | \
                                  (df['Third_label'] == self.abnormal_diagnose)].reset_index(drop=True)
        
        self.norm_pairs_count = len(self.normal_df) # Amount of pairs where each ECG with normal label is used once
        self.abnorm_pairs_count = len(self.abnormal_df) # Amount of pairs where each ECG with abnormal label is used once

        
    def __getitem__(self, index):

        # Getting pairs with normal label (labels are same)
        if index < self.norm_pairs_count:
            
            f_index = index
            s_index = (index + 1) % self.norm_pairs_count
            
            print(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')
            print(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][s_index]}.mat')

            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][s_index]}.mat')['ECG']
            label = 1.

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32),
                ), torch.as_tensor((label), dtype=torch.float32)
        
        else: index -= self.norm_pairs_count

        
        # Getting pairs with abnormal label (labels are same)
        if index < self.abnorm_pairs_count:

            f_index = index
            s_index = (index + 1) % self.abnorm_pairs_count
            
            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.abnormal_df["Recording"][f_index]}.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.abnormal_df["Recording"][s_index]}.mat')['ECG']
            label = 1.

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32),
                ), torch.as_tensor((label), dtype=torch.float32)
        
        else: index -= self.abnorm_pairs_count
        

        # Getting pairs with normal and abnormal labels (labels are different)
        if index < self.norm_pairs_count:
            
            f_index = index
            s_index = np.random.randint(0, self.abnorm_pairs_count)
            
            ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][f_index]}.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.abnormal_df["Recording"][s_index]}.mat')['ECG']
            label = 0.

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32),
                ), torch.as_tensor((label), dtype=torch.float32)
        
        else: index -= self.norm_pairs_count
        

        # Getting pairs with normal and abnormal labels (labels are different)
        f_index = index
        s_index = random.randint(0, self.norm_pairs_count - 1)
        
        ecg1 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.abnormal_df["Recording"][f_index]}.mat')['ECG']
        ecg2 = scipy.io.loadmat(f'Data\ChineseDataset\PreparedDataset_Noisy{self.path_append}\{self.normal_df["Recording"][s_index]}.mat')['ECG']
        label = 0.

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32),
            ), torch.as_tensor((label), dtype=torch.float32)
    

    def __len__(self):
        return  self.norm_pairs_count * 2 + self.abnorm_pairs_count * 2