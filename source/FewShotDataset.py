import os
import pandas as pd
import scipy

classes_to_classify = [59118001, 426783006, 111975006, 89792004, 67741000119109] # May be random

class FewShotDataset():

    def __init__(self, shot=3):
        self.shot = shot

        if not os.path.exists('GeorgiaDataset\META_DATA.csv'):
            self.gather_dataset_info()
        
        self.df = pd.read_csv('GeorgiaDataset\META_DATA.csv', index_col='diagnosis')

    def get_train_data(self):
        ECGs = []
        for diag in classes_to_classify:
            for i in range(0, self.shot):
                ECGs.append(scipy.io.loadmat(f'GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :2900])
        # print(ECGs)
        diagnoses = []
        for i in range(len(classes_to_classify)):
            for j in range(self.shot):
                diagnoses.append(classes_to_classify[i])
        return diagnoses, ECGs


    def get_test_data(self):
        ECGs = []
        for diag in classes_to_classify:
            for i in range(self.shot, 10):
                ECGs.append(scipy.io.loadmat(f'GeorgiaDataset\{self.df.loc[diag][i]}')['val'][:, :2900])
        diagnoses = []
        for i in range(len(classes_to_classify)):
            for j in range(self.shot, 10):
                diagnoses.append(classes_to_classify[i])
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