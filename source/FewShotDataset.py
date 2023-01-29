import random
import os
import pandas as pd
import numpy as np
import torch
import scipy.io
from PreprocessingFilters import filter1
from NoisyDataset import NoisyPairsDataset as NS_dataset
import math


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# classes_to_classify = [59118001, 426783006, 111975006, 89792004, 67741000119109] # May be random
classes_to_classify = [1, 2, 3, 5, 6]

class FewShotDataset():

    def __init__(self, shot=3):
        self.shot = shot

        if not os.path.exists('GeorgiaDataset\META_DATA.csv'):
            self.gather_dataset_info()
        
        # self.df = pd.read_csv('GeorgiaDataset\META_DATA.csv', index_col='diagnosis')
        self.df = pd.read_csv('ChineseDataset\REFERENCE.csv')

        if not os.path.exists(f'FewShot(Shot={self.shot}\\train)'):
            os.makedirs(f'FewShot(Shot={self.shot})\\train', exist_ok=True)
        if not os.path.exists(f'FewShot(Shot={self.shot}\\test)'):
            os.makedirs(f'FewShot(Shot={self.shot})\\test', exist_ok=True)

    def get_train_data(self):

        ECGs = []
        for diag in classes_to_classify:
            df_with_class = self.df.loc[(self.df['First_label'] == diag) & (self.df['Recording'] <= 'A2000')]
            for i in range(0, self.shot):
                # ECGs.append(scipy.io.loadmat(f'GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :3000])
                ECGs.append(scipy.io.loadmat(f'ChineseDataset\TrainingSet1\{df_with_class.iloc[i]["Recording"]}.mat')['ECG'][0][0][2][:, :3000])
        diagnoses = []
        for diag in classes_to_classify:
            for j in range(0, self.shot):
                diagnoses.append(diag)


        if (len(os.listdir(f'FewShot(Shot={self.shot})\\train')) == 0):
            ECGs = self.prepare_ECG(ECGs)
            self.save_to_dir(f'FewShot(Shot={self.shot})\\train', ECGs)
        else:
            ECGs = self.read_from_dir(f'FewShot(Shot={self.shot})\\train')


        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i], dtype=torch.float32)

        
        return diagnoses, ECGs


    def get_test_data(self):

        ECGs = []
        for diag in classes_to_classify:
            df_with_class = self.df.loc[(self.df['First_label'] == diag) & (self.df['Recording'] <= 'A2000')]
            for i in range(self.shot, 150):
                # ECGs.append(scipy.io.loadmat(f'GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :3000])
                ECGs.append(scipy.io.loadmat(f'ChineseDataset\TrainingSet1\{df_with_class.iloc[i]["Recording"]}.mat')['ECG'][0][0][2][:, :3000])
        diagnoses = []
        for diag in classes_to_classify:
            for j in range(self.shot, 150):
                diagnoses.append(diag)


        if (len(os.listdir(f'FewShot(Shot={self.shot})\\test')) == 0):
            ECGs = self.prepare_ECG(ECGs)
            self.save_to_dir(f'FewShot(Shot={self.shot})\\test', ECGs)
        else:
            ECGs = self.read_from_dir(f'FewShot(Shot={self.shot})\\test')


        for i in range(len(ECGs)):
            ECGs[i] = torch.as_tensor(ECGs[i], dtype=torch.float32)


        return diagnoses, ECGs

    def gather_dataset_info(self):
        ECG_META = {}

        for i in range(1, 10345):
            file = open(f'GeorgiaDataset\E{str(i).zfill(5)}.hea', 'r')
            loc_labels = self.get_labels(file.read())
            for label in loc_labels:
                if label not in ECG_META.keys():
                    ECG_META[label] = []
                ECG_META[label].append(f'E{str(i).zfill(5)}.mat')
            file.close()    

        ECG_META = self.pad_dict_list(ECG_META, '')

        print(ECG_META.keys())

        df = pd.DataFrame.from_dict(ECG_META).transpose()
        pd.DataFrame.to_csv(df, 'GeorgiaDataset\META_DATA.csv', index_label='diagnosis')

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
        # Filtering
        for i in range(len(ECGs)):
            ECGs[i] = self.filter_ecg(ECGs[i])
            self.print_progressBar(i+1, len(ECGs), prefix='Filtering ECG: ', length=50)

        means, stds = self.get_channel_means_stds()
        print(means)
        print(stds)
        # means = [0.0016, 0.0003, 0.0006, 0.0009, 0.0017, 0.0005, -0.0020, -0.0008, 0.0005, 0.0009, -1.4044e-05, 0.0003]
        # stds = [1.0365105173338283, 1.0212978097844168, 1.028844629063083, 1.0287473539964986, 1.0416054262597099, 1.0228705546648325, 1.021233960122595, 1.0075333082879048, 1.0084018610427552, 1.021604512664429, 1.0291869595234864, 1.0514313909410649]
        for i in range(len(ECGs)):
            new_ecg = []
            for k in range(12):
                if np.std(ECGs[i][k]) < 1e-8: std = 1
                else: std = np.std(ECGs[i][k])
                new_ecg.append((ECGs[i][k] - np.mean(ECGs[i][k])) / std)
            
            new_ecg = np.array(new_ecg)

            for k in range(12):
                new_ecg[k] = new_ecg[k] * stds[k] + means[k]

            ECGs[i] = new_ecg

            self.print_progressBar(i+1, len(ECGs), prefix='Normalizing ECG: ', length=50)

        return ECGs

    def get_channel_means_stds(self):
        dataset = NS_dataset()
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
            stds.append(math.sqrt(squared_sum / (counter - 1)))

        return means, stds

    def filter_ecg(self, ekg):
        struct1 = np.ones((ekg.shape[0], 6)) / 5
        struct2 = np.ones((ekg.shape[0], 45)) / 5
        data = filter1(ekg, struct1, struct2)[:, 50:-50]
        return data

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

    def save_to_dir(self, path, ECGs):
        for i, ECG in enumerate(ECGs):
            scipy.io.savemat(f'{path}\{i}.mat', {'ECG': ECG})

    def read_from_dir(self, path):
        ECGs = []
        for file in os.listdir(path):
            ECGs.append(scipy.io.loadmat(f'{path}\{file}')['ECG'])
        return ECGs