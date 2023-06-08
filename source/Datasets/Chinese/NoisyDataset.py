import os
from torch.utils.data import Dataset
from Datasets.Chinese.NoisyDatasetPreprocessing import prepare_dataset
import scipy
import torch
import random
import numpy as np
import pandas as pd

class NoisyPairsDataset(Dataset):
    
    def __init__(self, 
                 labels = [8, 9],
                 samples_per_element=5, 
                 folder='Train'):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        # Preparing dataset #########################################################################
        self.folder = folder
        if not self.is_data_ready(): prepare_dataset(f'Data/ChineseDataset')
        #############################################################################################

        self.NORMAL_LABEL = 1
        assert samples_per_element % len(labels) == 0, '\'samples_per_element\' should be multiple of \'labels\' length'
        self.labels = labels
        

        df = pd.read_csv(f'Data/ChineseDataset/{folder}/LOCAL_REFERENCE.csv', delimiter=',')
        # DATA_TYPES = ['NormFilteredECG', 'NormECG']
        DATA_TYPES = ['NormFilteredECG']


        self.normal_data = []
        normal_df = df.loc[
            (df['First_label'] == self.NORMAL_LABEL)
        ].reset_index(drop=True)

        for DATA_TYPE in DATA_TYPES:
            for i in range(len(normal_df)):
                self.normal_data.append(scipy.io.loadmat(f'Data/ChineseDataset/{self.folder}/{DATA_TYPE}/{normal_df["Recording"][i]}.mat')['ECG'])


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
                    abnormal_d.append(scipy.io.loadmat(f'Data/ChineseDataset/{self.folder}/{DATA_TYPE}/{abnormal_df["Recording"][i]}.mat')['ECG'])

            self.abnormal_data.append(abnormal_d)


        self.samples_per_normal = samples_per_element * len(self.abnormal_data)
        self.samples_per_abnormal = samples_per_element

        # These indices are used in order to guarantee the selection of the same ECGs in pairs to normal one in each EPOCH
        self.normal_indices = np.random.choice(len(self.normal_data), len(self.normal_data), replace=False)
        self.normal_indices = np.tile(self.normal_indices, self.samples_per_normal)
        self.abnormal_indices = [np.random.choice(len(abn_d), len(abn_d), replace=False) for abn_d in self.abnormal_data]

        self.ds_len = int(len(self.normal_data) * self.samples_per_normal +                             # For each normal ECG {samples} amount of random normal ECGs
                          len(self.normal_data) * len(self.abnormal_data) * self.samples_per_abnormal)  # For each normal ECG {samples / len(abnormal_data)} amount of random abnormal ECG from each abnormal label
        
    def __getitem__(self, index):

        # Pairs with equal normal labels
        if index < len(self.normal_data) * self.samples_per_normal:

            f_index = index // self.samples_per_normal
            s_index = self.normal_indices[index]

            return (
                    torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                    torch.as_tensor(self.normal_data[s_index], dtype=torch.float32),
                ), torch.as_tensor((1.), dtype=torch.float32)
            
        else: index -= len(self.normal_data) * self.samples_per_normal
        


        # Pairs with different labels (norm & abnorm)
        f_index = index // self.samples_per_normal
        abnormal_d = self.abnormal_data[index % len(self.abnormal_data)]
        indices = self.abnormal_indices[index % len(self.abnormal_data)]
        s_index = indices[index % len(indices)]

        return (
                torch.as_tensor(self.normal_data[f_index], dtype=torch.float32),
                torch.as_tensor(abnormal_d[s_index], dtype=torch.float32),
            ), torch.as_tensor((0.), dtype=torch.float32)

    def __len__(self):
        return  self.ds_len

    def is_data_ready(self):
        return os.path.exists(f'Data/ChineseDataset/{self.folder}/NormFilteredECG') \
            and os.path.exists(f'Data/ChineseDataset/{self.folder}/NormECG') \
                and len(os.listdir(f'Data/ChineseDataset/{self.folder}/NormFilteredECG')) != 0 \
                    and len(os.listdir(f'Data/ChineseDataset/{self.folder}/NormECG')) != 0