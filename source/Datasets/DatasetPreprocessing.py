import pandas as pd
import scipy.io
import numpy as np
from Filtering.PreprocessingFilters import filter1
import math

ECG_MAX_LEN = 15000


# USE NOLY FOR CHINESE DATASET!!!!
def prepare_dataset(fill_with_type, path='Data\ChineseDataset\\'):

    df = pd.read_csv(f'{path}\REFERENCE.csv', delimiter=',')
    df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)

    total_data = []


    print("Starting filtering...")


    for i in range(2000):
        ecg = scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2][:, 100:-100]
        
        if fill_with_type == 'mean':
            ecg = fill_with_mean(ecg)
        elif fill_with_type == 'cyclic_repeat':
            ecg = fill_with_same(ecg)
        else:
            raise Exception(f'Don\'t know what is {fill_with_type}. Try to use \'mean\' or \'cyclic_repeat\'')

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


    print("Filtering done! Starting channel-wise ECG normalization...")


    ### Channel-wise normalization
    channel_means, channel_stds = get_channel_means_stds(total_data)

    for i, recording in enumerate(total_data):
        for j in range(12):
            recording[0][j] = (recording[0][j] - channel_means[j]) / channel_stds[j]
        print_progressBar(i+1, 2000, prefix='Normalizing ECG:', length=50)


    print(f"Normaization done! Saving data to {path}PreparedDataset_{fill_with_type}\\")


    for i, recording in enumerate(total_data):
        scipy.io.savemat(f'{path}PreparedDataset_{fill_with_type}\{recording[1]}.mat', {'ECG': recording[0]})
        print_progressBar(i+1, 2000, prefix='Saving:', length=50)

    print("Normalization done! Dataset preparation complete!")

    
def fill_with_mean(ecg):
    mean = []
    if (ECG_MAX_LEN - ecg.shape[1]) > 0:
        for i in range(12):
            mean.append(np.full(ECG_MAX_LEN - ecg.shape[1], np.mean(ecg[i])))
        ecg = np.column_stack([ecg, mean])
    else:
        ecg = ecg[:, :ECG_MAX_LEN]
    return ecg

def fill_with_same(ecg):
    if (ECG_MAX_LEN - ecg.shape[1]) > 0:
        times_to_duplucate = ECG_MAX_LEN / ecg.shape[1]
        ecg = np.tile(ecg, math.ceil(times_to_duplucate))[:, :ECG_MAX_LEN]
    else:
        ecg = ecg[:, :ECG_MAX_LEN]
    return ecg
        
        
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

    # print('means: ', means)
    # print('stds: ', stds)

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