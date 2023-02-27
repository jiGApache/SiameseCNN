import os
import pandas as pd
import scipy
from Filtering.Neurokit2Filters import filter_ecg
# from Filtering.PreprocessingFilters import filter_ecg
import numpy as np
import math

np.random.seed(42)

FRAGMENT_SIZE = 3000    # or KERNEL_SIZE
STEP_SIZE = 1500        # or STRIDE
df = pd.read_csv('Data\ChineseDataset\REFERENCE.csv', delimiter=',')
df = df.loc[df['Recording'] <= 'A2000'].reset_index(drop=True)
dataset_size = len(df)
total_data = []

def prepare_dataset(path='Data\ChineseDataset\PreparedDataset_Noisy\\'):


    if not os.path.exists('Data\ChineseDataset\FilteredECG') or len(os.listdir('Data\ChineseDataset\FilteredECG')) == 0:
        os.mkdir('Data\ChineseDataset\FilteredECG')
        for i in range(dataset_size):
            ecg = scipy.io.loadmat('Data\ChineseDataset\TrainingSet1\\' + df['Recording'][i] + '.mat')['ECG'][0][0][2]
        
            ### Filtering EKG
            ecg = filter_ecg(ecg)

            scipy.io.savemat(f'Data\ChineseDataset\FilteredECG\{df["Recording"][i]}.mat', {'ECG': ecg})

            recording = [ecg[:, 100:100+FRAGMENT_SIZE], df['Recording'][i]]        
            total_data.append(recording)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)


    else:
        for i in range(dataset_size):
            ecg = scipy.io.loadmat(f'Data\ChineseDataset\FilteredECG\{df["Recording"][i]}.mat')['ECG']

            recording = [ecg, df['Recording'][i]]
            total_data.append(recording)

            print_progressBar(i+1, dataset_size, prefix='Filtering ECG:', length=50)

    print("Filtering done! Starting channel-wise ECG normalization...")


    ## Channel-wise normalization
    channel_means, channel_stds = get_channel_means_stds(total_data)
    for i, recording in enumerate(total_data):
        for j in range(12):
            recording[0][j] = (recording[0][j] - channel_means[j]) / channel_stds[j]
        print_progressBar(i+1, dataset_size, prefix='Normalizing ECG:', length=50)

    print(f"Normaization done! Saving data to {path}")


    for i, recording in enumerate(total_data):

        noise = get_baseline_noise()

        if 'Rolled' in path:
            roll = np.random.randint(-1500, 1500)
            scipy.io.savemat(f'{path}{recording[1]}.mat', {'ECG': np.roll(recording[0] + noise, roll, axis=1)})
        else:
            scipy.io.savemat(f'{path}{recording[1]}.mat', {'ECG': recording[0] + noise})

        print_progressBar(i+1, dataset_size, prefix='Saving:', length=50)

    print("Dataset preparation complete!")


def get_channel_means_stds(total_data):
    
    ECGs = []
    for i in range(len(total_data)):
        ECGs.append(total_data[i][0])

    ECGs = np.asarray(ECGs)
    means = ECGs.mean(axis=(0,2))
    stds = ECGs.std(axis=(0,2))

    del ECGs

    return means, stds

def get_baseline_noise():
    
    #Baseline wander
    L = FRAGMENT_SIZE
    x = np.linspace(0, L, L)
    A = np.random.uniform(0.1, 1.)
    T = 2 * L
    PHI = np.random.uniform(0, 2 * math.pi)
    wander = []
    for j in x:
        wander.append(A * np.cos(2 * math.pi * (j/T) + PHI))
    wander = np.asarray(wander)
    
    return np.tile(wander, (12,1))

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