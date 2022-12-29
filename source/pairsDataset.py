import random
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from DatasetPreprocessing import prepare_dataset

class PairsDataset(Dataset):
    
    def __init__(self, same_size=True):
        self.EKG_MAX_LEN = 15000
        self.same_size = same_size
        # self.normalize = normalize
        # self.filter = filter
        # self.means = [5.6671288368373204e-08, -5.672094472486019e-08, -1.1381123812568519e-07,
        #          -7.73628575187182e-10, 8.544064961353723e-08, -8.466800578420468e-08,
        #          -5.644898281745803e-08, -3.3201897366838757e-07, -6.639807663731727e-08,
        #          -1.9771499946997733e-08, -3.3253429074075554e-08, 1.487236435452322e-07]
        # self.stds = [0.2305271687030844, 0.24780370485706876, 0.23155043161905942,
        #         0.22074153961314927, 0.20708526280758174, 0.22153766144293813,
        #         0.353942949952694, 0.3942397032518631, 0.4228515959530688,
        #         0.436324876078121, 0.47316252072611537, 0.5328047065188085]

        if (not os.path.exists('ChineseDataset\PreparedDataset')) or (len(os.listdir('ChineseDataset\PreparedDataset')) == 0):
            prepare_dataset()

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

    # def filter_ekg(self, mat_file):
    #     struct1 = np.ones((mat_file.shape[0], 6)) / 5
    #     struct2 = np.ones((mat_file.shape[0], 45)) / 5
    #     data = filter1(mat_file, struct1, struct2)[:, 100:-100]
    #     return data

    def to_same_size(self, ekg):
        mean = []
        if (self.EKG_MAX_LEN - ekg.shape[1]) > 0:
            for i in range(12):
                mean.append(np.full(self.EKG_MAX_LEN - ekg.shape[1], np.mean(ekg[i])))
            ekg = np.column_stack([ekg, mean])
        else:
            ekg = ekg[:, :self.EKG_MAX_LEN]
        return ekg


    def getEqPair(self, df, index):
        
        if (index >= 0) and (index < (df.shape[0])):
            counter = f_el_pos = 0
            s_el_pos = 1
            while (counter != index):
                f_el_pos += 1
                s_el_pos = (s_el_pos + 1) % df.shape[0]
                counter += 1
            
            ekg1 = np.genfromtxt(f'ChineseDataset\PreparedDataset\{df["Recording"][f_el_pos]}.csv')
            ekg2 = np.genfromtxt(f'ChineseDataset\PreparedDataset\{df["Recording"][s_el_pos]}.csv')
            # ekg1 = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df['Recording'][f_el_pos] + '.mat')['ECG'][0][0][2]
            # ekg2 = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df['Recording'][s_el_pos] + '.mat')['ECG'][0][0][2]
            
            ### Filtering EKG
            # if self.filter:
            #     ekg1 = self.filter_ekg(ekg1)
            #     ekg2 = self.filter_ekg(ekg2)
            #################

            ### Normalization
            # if self.normalize:
            #     for i in range(12):
            #         ekg1[i] = (ekg1[i] - self.means[i]) / self.stds[i]
            #         ekg2[i] = (ekg2[i] - self.means[i]) / self.stds[i]
            #################

            ### EKG to EKG_MAX_LEN
            if self.same_size:
                ekg1 = self.to_same_size(ekg1)
                ekg2 = self.to_same_size(ekg2)
            ######################

            return (
                torch.FloatTensor(ekg1),
                torch.FloatTensor(ekg2),
            ), torch.tensor((1.))
        else: index -= df.shape[0]
        
        return index

    

    def getDifPair(self, df1, df2, index):

        if (index >= 0) and (index < df1.shape[0]):
            rand_item = random.randint(0, (df2.shape[0]) - 1)

            random_ekg = np.genfromtxt(f'ChineseDataset\PreparedDataset\{df2["Recording"][rand_item]}.csv')
            ekg = np.genfromtxt(f'ChineseDataset\PreparedDataset\{df1["Recording"][index]}.csv')
            # random_ekg = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df2['Recording'][rand_item] + '.mat')['ECG'][0][0][2]
            # ekg = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df1['Recording'][index] + '.mat')['ECG'][0][0][2]

            ### Filtering EKG
            # if self.filter:
            #     random_ekg = self.filter_ekg(random_ekg)
            #     ekg = self.filter_ekg(ekg)
            #################

            ### Normalization
            # if self.normalize:
            #     for i in range(12):
            #         random_ekg[i] = (random_ekg[i] - self.means[i]) / self.stds[i]
            #         ekg[i] = (ekg[i] - self.means[i]) / self.stds[i]
            #################

            ### EKG to EKG_MAX_LEN
            if self.same_size:
                random_ekg = self.to_same_size(random_ekg)
                ekg = self.to_same_size(ekg)
            ######################

            return (
                torch.FloatTensor(ekg),
                torch.FloatTensor(random_ekg)
            ), torch.tensor((0.))
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

        # return ('UNKNOWN', 'UNKNOWN'), torch.tensor([2.])
        raise IndexError('Dataset index is out of range')



    def __len__(self):
        
        total_pairs = 0

        for df in self.with_df:
            total_pairs += df.shape[0]

        return total_pairs * 2