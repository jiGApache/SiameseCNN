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
    
    def __init__(self, labels = [3, 5], folder='Train'):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Preparing dataset #########################################################################
        self.folder = folder
        if not self.is_data_ready(): prepare_dataset(f'Data\ChineseDataset')
        #############################################################################################

        self.NORMAL_LABEL = 1
        self.labels = labels

        df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
        if self.folder == 'Train':
            df = df.loc[df['Recording'] <= 'A4470']
        else:
            df = df.loc[df['Recording'] >= 'A4471']
        


        # DATA_TYPES = ['NormFilteredECG', 'NormECG']
        DATA_TYPES = ['NormFilteredECG']



        self.normal_data = []
        normal_df = df.loc[
            (df['First_label'] == self.NORMAL_LABEL)
        ].reset_index(drop=True)

        for DATA_TYPE in DATA_TYPES:
            for i in range(len(normal_df)):
                self.normal_data.append(scipy.io.loadmat(f'Data\ChineseDataset\{self.folder}\{DATA_TYPE}\{normal_df["Recording"][i]}.mat')['ECG'])



        self.abnormal_data = []
        for label in labels:

            abnormal_d = []
            abnormal_df = df.loc[
                (df['First_label'] == label) & \
                (np.isnan(df['Second_label'])) & \
                (np.isnan(df['Third_label']))
            ].reset_index(drop=True)

            for DATA_TYPE in DATA_TYPES:
                for i in range(len(abnormal_df)):
                    abnormal_d.append(scipy.io.loadmat(f'Data\ChineseDataset\{self.folder}\{DATA_TYPE}\{abnormal_df["Recording"][i]}.mat')['ECG'])

            self.abnormal_data.append(abnormal_d)

        self.ds_len = 0
        self.ds_len += len(self.normal_data) * 10 + len(self.normal_data) * len(self.abnormal_data) * 5
        
    def __getitem__(self, index):

        # Pairs with equal normal labels
        if index < len(self.normal_data) * 10:

            f_index = index // 10
            s_index = np.random.randint(0, len(self.normal_data))

            return (
                    torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                    torch.as_tensor(self.normal_data[s_index], dtype=torch.float32),
                ), torch.as_tensor((1.), dtype=torch.float32)
            
        else: index -= len(self.normal_data) * 10
        


        # Pairs with different labels (norm & abnorm)
        f_index = index // (len(self.abnormal_data) * 5)
        abnormal_d = self.abnormal_data[index % len(self.abnormal_data)]
        s_index = np.random.randint(0, len(abnormal_d))

        return (
                torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                torch.as_tensor(abnormal_d[s_index], dtype=torch.float32),
            ), torch.as_tensor((0.), dtype=torch.float32)

    def __len__(self):
        return  self.ds_len

    def is_data_ready(self):
        return os.path.exists(f'Data\ChineseDataset\{self.folder}\\NormFilteredECG') \
            and os.path.exists(f'Data\ChineseDataset\{self.folder}\\NormECG') \
                and len(os.listdir(f'Data\ChineseDataset\{self.folder}\\NormFilteredECG')) != 0 \
                    and len(os.listdir(f'Data\ChineseDataset\{self.folder}\\NormECG')) != 0