from torch.utils.data import Dataset
import random
import numpy as np
import os
import pandas as pd
import scipy.io
import torch
from  Datasets.Physionet.NoisyDatasetPreprocessing import prepare_dataset

class PairsDataset(Dataset):
    def __init__(self, 
                 samples_per_element=5,
                 folder='Train'):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)


        df = pd.read_csv(f'Data/PTB-XL/{folder}/LOCAL_REFERENCE.csv', index_col=None)
        # DATA_TYPES = ['NormFilteredECG', 'NormECG']
        DATA_TYPES = ['NormFilteredECG']
        self.NORMAL_COL = 'NORM'
        self.abnorm_cols = ['STTC', 'MI', 'HYP', 'CD']

        # Preparing dataset #########################################################################
        self.folder = folder
        if not self.is_data_ready(): prepare_dataset(f'Data/PTB-XL/')
        #############################################################################################


        self.normal_data = []
        normal_df = df.loc[
            (df[self.NORMAL_COL] == 1) & \
            (df['STTC'] == 0) & \
            (df['MI'] == 0) & \
            (df['HYP'] == 0) & \
            (df['CD'] == 0)
        ].reset_index(drop=True)

        for DATA_TYPE in DATA_TYPES:
            for i in range(len(normal_df)):
                self.normal_data.append(f'{str(normal_df["ecg_id"][i]).zfill(5)}.mat')

        self.abnormal_data = []
        for col in self.abnorm_cols:

            abnormal_d = []
            abnormal_df = df.loc[
                (df[self.NORMAL_COL] == 0) & \
                (df[col] == 1)
            ].reset_index(drop=True)

            for DATA_TYPE in DATA_TYPES:
                for i in range(len(abnormal_df)):
                    abnormal_d.append(f'{str(abnormal_df["ecg_id"][i]).zfill(5)}.mat')

            self.abnormal_data.append(abnormal_d)



        self.samples_per_normal = samples_per_element * len(self.abnormal_data)
        self.samples_per_abnormal = samples_per_element

        # These indices are used in order to guarantee the selection of the same ECGs in pairs to normal one in each EPOCH
        self.normal_indices = np.random.choice(len(self.normal_data), len(self.normal_data), replace=False)
        self.normal_indices = np.tile(self.normal_indices, self.samples_per_normal)
        self.abnormal_indices = [np.random.choice(len(abn_d), len(abn_d), replace=False) for abn_d in self.abnormal_data]

        self.ds_len = int(len(self.normal_data) * self.samples_per_normal +                             # For each normal ECG {samples_per_normal} amount of random normal ECGs
                          len(self.normal_data) * len(self.abnormal_data) * self.samples_per_abnormal)  # For each normal ECG {len(abnormal_data) * samples_per_abnormal} amount of random abnormal ECG from each abnormal label


    def __getitem__(self, index):


        # Pairs with equal normal labels
        if index < len(self.normal_data) * self.samples_per_normal:
            
            f_index = index // self.samples_per_normal
            s_index = self.normal_indices[index]

            ecg1 = scipy.io.loadmat(f'Data/PTB-XL/{self.folder}/NormFilteredECG/{self.normal_data[f_index]}')['ECG']
            ecg2 = scipy.io.loadmat(f'Data/PTB-XL/{self.folder}/NormFilteredECG/{self.normal_data[s_index]}')['ECG']

            return (
                    torch.as_tensor(ecg1, dtype=torch.float32),
                    torch.as_tensor(ecg2, dtype=torch.float32)
                ), torch.as_tensor((1.), dtype=torch.float32)
            
        else: index -= len(self.normal_data) * self.samples_per_normal
        


        # Pairs with different labels (norm & abnorm)
        f_index = index // self.samples_per_normal
        abnormal_d = self.abnormal_data[index % len(self.abnormal_data)]
        indices = self.abnormal_indices[index % len(self.abnormal_data)]
        s_index = indices[index % len(indices)]

        ecg1 = scipy.io.loadmat(f'Data/PTB-XL/{self.folder}/NormFilteredECG/{self.normal_data[f_index]}')['ECG']
        ecg2 = scipy.io.loadmat(f'Data/PTB-XL/{self.folder}/NormFilteredECG/{abnormal_d[s_index]}')['ECG']

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32)
            ), torch.as_tensor((0.), dtype=torch.float32)

    def __len__(self):
        return self.ds_len
    

    def is_data_ready(self):
        return os.path.exists(f'Data/PTB-XL/{self.folder}/NormFilteredECG') \
            and os.path.exists(f'Data/PTB-XL/{self.folder}/NormECG') \
                and len(os.listdir(f'Data/PTB-XL/{self.folder}/NormFilteredECG')) != 0 \
                    and len(os.listdir(f'Data/PTB-XL/{self.folder}/NormECG')) != 0