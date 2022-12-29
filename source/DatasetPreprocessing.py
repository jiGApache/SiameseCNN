import pandas as pd
import scipy.io
import numpy as np
from PreprocessingFilters import filter1
import math

def prepare_dataset(path='ChineseDataset\PreparedDataset\\'):

    df = pd.read_csv('ChineseDataset\REFERENCE.csv', delimiter=',')
    df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)

    total_data = []

    print("Starting filtering...")

    for i in range(2000):
        ecg = scipy.io.loadmat('ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2]
        
        ### Filtering EKG
        ecg = filter_ekg(ecg)

        ### ECG-wise normalization
        for j in range(12):
            channel_std = ecg[j].std()
            if abs(channel_std - 0.0) <= 1e-08: channel_std = 1
            ecg[j] = (ecg[j] - ecg[j].mean()) / channel_std
        
        recording = [ecg, df['Recording'][i]]        

        total_data.append(recording)

        print_progressBar(i+1, 2000, prefix='Filtering ECG:', length=50)

    print("Filtering done!\nStarting channel-wise ECG normalization...")

    ### Channel-wise normalization
    channel_means, channel_stds = get_channel_means_stds(total_data)
    for i, recording in enumerate(total_data):
        for j in range(12):
            recording[0][j] = (recording[0][j] - channel_means[j]) / channel_stds[j]
        np.savetxt(f'{path}{recording[1]}.csv', recording[0])
        print_progressBar(i+1, 2000, prefix='Normalizing ECG:', length=50)

    print("Normalization done!\nDataset praparation complete!")

    
        
        
def get_channel_means_stds(total_data):

    channels_of_12 = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for recording in total_data:
        for j in range(12):
            channels_of_12[j].append(recording[0][j])

    means = []
    stds = []
    for channel in channels_of_12:
        
        counter = 0
        regular_sum = 0
        squared_sum = 0

        for element in channel:
            counter += len(element)
            regular_sum += sum(element)
            squared_sum += sum(pow(element - regular_sum / counter, 2))

        means.append(regular_sum / counter)
        stds.append(math.sqrt(squared_sum / counter))

    print('means: ', means)
    print('stds: ', stds)

    return means, stds

def filter_ekg(ekg):
    struct1 = np.ones((ekg.shape[0], 6)) / 5
    struct2 = np.ones((ekg.shape[0], 45)) / 5
    data = filter1(ekg, struct1, struct2)[:, 100:-100]
    return data

def print_progressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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