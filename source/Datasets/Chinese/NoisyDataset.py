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
        if not self.is_data_ready(): prepare_dataset(f'Data\ChineseDataset')
        #############################################################################################

        self.NORMAL_LABEL = 1
        self.labels = labels

        df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
        df = df.loc[df['Recording'] <= 'A4470']
        


        DATA_TYPES = ['NormFilteredECG', 'NormECG']



        self.normal_data = []
        normal_df = df.loc[
                (df['First_label'] == self.NORMAL_LABEL) | \
                (df['Second_label'] == self.NORMAL_LABEL) | \
                (df['Third_label'] == self.NORMAL_LABEL)
            ].reset_index(drop=True)

        for DATA_TYPE in DATA_TYPES:
            for i in range(len(normal_df)):
                self.normal_data.append(scipy.io.loadmat(f'Data\ChineseDataset\Train\{DATA_TYPE}\{normal_df["Recording"][i]}.mat')['ECG'])



        self.abnormal_data = []
        for label in labels:

            abnormal_d = []
            abnormal_df = df.loc[
                (df['First_label'] == label) | \
                (df['Second_label'] == label) | \
                (df['Third_label'] == label)
            ].reset_index(drop=True)

            for DATA_TYPE in DATA_TYPES:
                for i in range(len(abnormal_df)):
                    abnormal_d.append(scipy.io.loadmat(f'Data\ChineseDataset\Train\{DATA_TYPE}\{abnormal_df["Recording"][i]}.mat')['ECG'])

            self.abnormal_data.append(abnormal_d)

        self.ds_len = 0
        self.ds_len += len(self.normal_data) * 2 + len(self.normal_data) * len(self.abnormal_data)
        
    def __getitem__(self, index):

        # Pairs with equal normal labels
        if index < len(self.normal_data) * 2:

            f_index = index // 2
            s_index = np.random.randint(0, len(self.normal_data))

            return (
                    torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                    torch.as_tensor(self.normal_data[s_index], dtype=torch.float32),
                ), torch.as_tensor((1.), dtype=torch.float32)
            
        else: index -= len(self.normal_data) * 2      
        


        # Pairs with different labels (norm & abnorm)
        f_index = index // len(self.abnormal_data)
        abnormal_d = self.abnormal_data[index % len(self.abnormal_data)]
        s_index = np.random.randint(0, len(abnormal_d))

        return (
                torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                torch.as_tensor(abnormal_d[s_index], dtype=torch.float32),
            ), torch.as_tensor((0.), dtype=torch.float32)

    def __len__(self):
        return  self.ds_len

    def is_data_ready(self):
        return os.path.exists('Data\ChineseDataset\Train\\NormFilteredECG') \
            and os.path.exists('Data\ChineseDataset\Train\\NormECG') \
                and len(os.listdir('Data\ChineseDataset\Train\\NormFilteredECG')) != 0 \
                    and len(os.listdir('Data\ChineseDataset\Train\\NormECG')) != 0