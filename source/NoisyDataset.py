import os
from torch.utils.data import Dataset
from NoisyDatasetPreprocessing import prepare_dataset
import scipy
import math
import torch
import random

class NoisyPairsDataset(Dataset):
    
    def __init__(self):

        if (not os.path.exists(f'ChineseDataset\PreparedDataset_Noisy')):
            os.mkdir(f'ChineseDataset\PreparedDataset_Noisy')
        if (len(os.listdir(f'ChineseDataset\PreparedDataset_Noisy')) == 0):
            prepare_dataset()
        
    def __getitem__(self, index):
        if index % 2 ==0:
            index = str(int(index/2 + 1)).zfill(4)
            ecg1 = scipy.io.loadmat(f'ChineseDataset\PreparedDataset_Noisy\A{index}_1_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'ChineseDataset\PreparedDataset_Noisy\A{index}_2_noisy.mat')['ECG']
            label = 0.
        else:
            index = str(math.ceil(index/2)).zfill(4)
            rand_index = str(random.randint(1, 2000)).zfill(4)
            while index != rand_index:
                rand_index = str(random.randint(1, 2000)).zfill(4)
            ecg1 = scipy.io.loadmat(f'ChineseDataset\PreparedDataset_Noisy\A{index}_1_clean.mat')['ECG']
            ecg2 = scipy.io.loadmat(f'ChineseDataset\PreparedDataset_Noisy\A{rand_index}_2_noisy.mat')['ECG']
            label = 1.

        return (
                torch.as_tensor(ecg1, dtype=torch.float32),
                torch.as_tensor(ecg2, dtype=torch.float32),
            ), torch.as_tensor((label), dtype=torch.float32)

    def __len__(self):
        return 4000