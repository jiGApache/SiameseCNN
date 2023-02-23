import random
import os
import pandas as pd
import numpy as np
import torch
import scipy.io
from Filtering.PreprocessingFilters import filter1
from Datasets.Chinese.NoisyDataset import NoisyPairsDataset as NS_dataset
import math
from torch.utils.data import Dataset


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# classes_to_classify = [59118001, 426783006, 111975006, 89792004, 67741000119109] # May be random
classes_to_classify = [1, 2, 3, 5, 6]

class FewShotDataset(Dataset):

    def __init__(self, shot=3):
        self.shot = shot

        # if not os.path.exists('Data\GeorgiaDataset\META_DATA.csv'):
        #     self.gather_dataset_info()
        
        # self.df = pd.read_csv('Data\GeorgiaDataset\META_DATA.csv', index_col='diagnosis')
        self.df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv')

    def get_train_data(self):

        ECGs = []
        for diag in classes_to_classify:
            df_with_class = self.df.loc[(self.df['First_label'] == diag) & (self.df['Recording'] <= 'A2000')]
            for i in range(0, self.shot):
                # ECGs.append(scipy.io.loadmat(f'Data\GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :3000])
                ECGs.append(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{df_with_class.iloc[i]["Recording"]}.mat')['ECG'][:, :2900])
        diagnoses = []
        for diag in classes_to_classify:
            for _ in range(0, self.shot):
                diagnoses.append(diag)

        ECGs = self.prepare_ECG(ECGs)

        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i][None, :, :], dtype=torch.float32)

        
        return diagnoses, ECGs

    def get_test_data(self):

        ECGs = []
        for diag in classes_to_classify:
            df_with_class = self.df.loc[(self.df['First_label'] == diag) & (self.df['Recording'] <= 'A2000')]
            for i in range(self.shot, 150):
                # ECGs.append(scipy.io.loadmat(f'Data\GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :3000])
                ECGs.append(scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{df_with_class.iloc[i]["Recording"]}.mat')['ECG'][:, :2900])
        diagnoses = []
        for diag in classes_to_classify:
            for _ in range(self.shot, 150):
                diagnoses.append(diag)

        ECGs = self.prepare_ECG(ECGs)

        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i][None, :, :], dtype=torch.float32)


        return diagnoses, ECGs

    def gather_dataset_info(self):
        ECG_META = {}

        for i in range(1, 10345):
            file = open(f'Data\GeorgiaDataset\E{str(i).zfill(5)}.hea', 'r')
            loc_labels = self.get_labels(file.read())
            for label in loc_labels:
                if label not in ECG_META.keys():
                    ECG_META[label] = []
                ECG_META[label].append(f'E{str(i).zfill(5)}.mat')
            file.close()    

        ECG_META = self.pad_dict_list(ECG_META, '')

        print(ECG_META.keys())

        df = pd.DataFrame.from_dict(ECG_META).transpose()
        pd.DataFrame.to_csv(df, 'Data\GeorgiaDataset\META_DATA.csv', index_label='diagnosis')

    def get_labels(header):
            labels = list()
            for l in header.split('\n'):
                if l.startswith('#Dx'):
                    try:
                        entries = l.split(': ')[1].split(',')
                        for entry in entries:
                            labels.append(entry.strip())
                    except:
                        pass
            return labels

    def pad_dict_list(dict_list, padel):
        lmax = 0
        for lname in dict_list.keys():
            lmax = max(lmax, len(dict_list[lname]))
        for lname in dict_list.keys():
            ll = len(dict_list[lname])
            if  ll < lmax:
                dict_list[lname] += [padel] * (lmax - ll)
        return dict_list

    def prepare_ECG(self, ECGs):

        # means, stds = self.get_channel_means_stds()
        # print(means)
        # print(stds)
        means = [5.99627363843412e-05, 2.397782159529932e-05, -2.9988930277035585e-05, -2.9961088525606906e-05, 5.105385735935625e-05, -2.870212523371087e-06, -7.207850672016848e-05, -4.1560294832370996e-05, 3.959292793974337e-05, 8.824617924109024e-05, 0.0001283318897487918, 7.791441335800445e-05]
        stds = [0.17314126577697408, 0.1909722425382492, 0.17436245266720773, 0.16158359746907913, 0.14613192801879915, 0.16186289945417426, 0.30696264143578034, 0.3423613932813311, 0.36725664334713926, 0.3727159624619992, 0.41711148904861833, 0.48217168988897435]
        for i in range(len(ECGs)):
            new_ecg = []
            for k in range(ECGs[i].shape[0]):
                new_ecg.append((ECGs[i][k] - means[k]) / stds[k])
            new_ecg = np.array(new_ecg)

            ECGs[i] = new_ecg

            self.print_progressBar(i+1, len(ECGs), prefix='Normalizing ECG: ', length=50)

        return ECGs

    def get_channel_means_stds(self):
        dataset = NS_dataset(WITH_ROLL=True)
        channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

        for i in range(0, len(dataset), 2):
            pair, _ = dataset.__getitem__(i)
            ecg_fragment = pair[0]
            for i in range(ecg_fragment.shape[0]):
                channels_of_12[i].append(ecg_fragment[i])

        means = []
        stds = []
        for channel in channels_of_12:
            
            counter = 0
            regular_sum = 0
            squared_sum = 0

            for element in channel:
                counter += len(element)
                regular_sum += sum(element)
            for element in channel:
                squared_sum += sum(pow(element - regular_sum / counter, 2))

            means.append(regular_sum / counter)
            stds.append(math.sqrt(squared_sum / counter))

        return means, stds

    def print_progressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()