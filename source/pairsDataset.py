import random
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from DatasetPreprocessing import prepare_dataset

class PairsDataset(Dataset):
    
    def __init__(self, device, fill_with_type):
        self.device = device
        self.fill_with_type = fill_with_type

        if (not os.path.exists(f'ChineseDataset\PreparedDataset_{fill_with_type}')):
            os.mkdir(f'ChineseDataset\PreparedDataset_{fill_with_type}')
        elif (len(os.listdir(f'ChineseDataset\PreparedDataset_{fill_with_type}')) == 0):
            prepare_dataset(fill_with_type)


        self.df = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')

        self.with_df = []
        self.without_df = []

        self.with_df.append(self.df.loc[((self.df['First_label'] == 1) | (self.df['Second_label'] == 1) | (self.df['Third_label'] == 1)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 2) | (self.df['Second_label'] == 2) | (self.df['Third_label'] == 2)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 3) | (self.df['Second_label'] == 3) | (self.df['Third_label'] == 3)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 4) | (self.df['Second_label'] == 4) | (self.df['Third_label'] == 4)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 5) | (self.df['Second_label'] == 5) | (self.df['Third_label'] == 5)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 6) | (self.df['Second_label'] == 6) | (self.df['Third_label'] == 6)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 7) | (self.df['Second_label'] == 7) | (self.df['Third_label'] == 7)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 8) | (self.df['Second_label'] == 8) | (self.df['Third_label'] == 8)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.with_df.append(self.df.loc[((self.df['First_label'] == 9) | (self.df['Second_label'] == 9) | (self.df['Third_label'] == 9)) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))

        self.without_df.append(self.df.loc[(self.df['First_label'] != 1) & (self.df['Second_label'] != 1) & (self.df['Third_label'] != 1) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 2) & (self.df['Second_label'] != 2) & (self.df['Third_label'] != 2) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 3) & (self.df['Second_label'] != 3) & (self.df['Third_label'] != 3) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 4) & (self.df['Second_label'] != 4) & (self.df['Third_label'] != 4) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 5) & (self.df['Second_label'] != 5) & (self.df['Third_label'] != 5) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 6) & (self.df['Second_label'] != 6) & (self.df['Third_label'] != 6) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 7) & (self.df['Second_label'] != 7) & (self.df['Third_label'] != 7) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 8) & (self.df['Second_label'] != 8) & (self.df['Third_label'] != 8) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))
        self.without_df.append(self.df.loc[(self.df['First_label'] != 9) & (self.df['Second_label'] != 9) & (self.df['Third_label'] != 9) & (self.df['Recording'] <= 'A2000')].reset_index(drop=True))

    def getEqPair(self, df, index):
        
        if (index >= 0) and (index < (df.shape[0])):
            counter = f_el_pos = 0
            s_el_pos = 1
            while (counter != index):
                f_el_pos += 1
                s_el_pos = (s_el_pos + 1) % df.shape[0]
                counter += 1
            
            ekg1 = np.genfromtxt(f'ChineseDataset\PreparedDataset_{self.fill_with_type}\{df["Recording"][f_el_pos]}.csv')
            ekg2 = np.genfromtxt(f'ChineseDataset\PreparedDataset_{self.fill_with_type}\{df["Recording"][s_el_pos]}.csv')

            return (
                torch.as_tensor(ekg1, dtype=torch.float32),#FloatTensor(ekg1),
                torch.as_tensor(ekg2, dtype=torch.float32),#FloatTensor(ekg2),
            ), torch.as_tensor((1.), dtype=torch.float32)
        else: index -= df.shape[0]
        
        return index

    

    def getDifPair(self, df1, df2, index):

        if (index >= 0) and (index < df1.shape[0]):
            rand_item = random.randint(0, (df2.shape[0]) - 1)

            random_ekg = np.genfromtxt(f'ChineseDataset\PreparedDataset_{self.fill_with_type}\{df2["Recording"][rand_item]}.csv')
            ekg = np.genfromtxt(f'ChineseDataset\PreparedDataset_{self.fill_with_type}\{df1["Recording"][index]}.csv')

            return (
                torch.as_tensor(ekg, dtype=torch.float32),#FloatTensor(ekg),
                torch.as_tensor(random_ekg, dtype=torch.float32),#FloatTensor(random_ekg)
            ), torch.as_tensor((0.), dtype=torch.float32)
        else:  index -= df1.shape[0]

        return index



    def __getitem__(self, index):

        for df in self.with_df:
            result = self.getEqPair(df, index)
            if type(result) == tuple:
                return result
            else: index = result

        for (with_df, without_df) in zip(self.with_df, self.without_df):
            result = self.getDifPair(with_df, without_df, index)
            if type(result) == tuple:
                return result
            else: index = result

        raise IndexError('Dataset index is out of range')



    def __len__(self):
        
        total_pairs = 0

        for df in self.with_df:
            total_pairs += df.shape[0]

        return total_pairs * 2