import os
import pandas as pd
import scipy
from Filtering.Neurokit2Filters import filter_ecg
# from Filtering.PreprocessingFilters import filter_ecg
import numpy as np

np.random.seed(42)
FRAGMENT_SIZE = 3000

def prepare_dataset(path='Data\ChineseDataset'):

    df = pd.read_csv(f'{path}\REFERENCE.csv', delimiter=',')
    train_df = df.loc[df['Recording'] <= 'A4470'].reset_index(drop=True)
    test_df = df.loc[df['Recording'] >= 'A4471'].reset_index(drop=True)
    total_data = []



    ### Filtering Train Data
    if not os.path.exists(f'{path}\Train\FilteredECG'): os.mkdir(f'{path}\Train\FilteredECG')
    if len(os.listdir(f'{path}\Train\FilteredECG')) == 0:
        for i in range(len(train_df)):
            ecg = scipy.io.loadmat(f'{path}\Train\InitialSet\{train_df["Recording"][i]}.mat')['ECG'][0][0][2]
        
            ecg = filter_ecg(ecg)[:, 100:FRAGMENT_SIZE+100]

            scipy.io.savemat(f'{path}\Train\FilteredECG\{train_df["Recording"][i]}.mat', {'ECG': ecg})

            recording = [ecg, train_df['Recording'][i]]        
            total_data.append(recording)

            print_progressBar(i+1, len(train_df), prefix='Filtering Train ECG:', length=50)
    else:
        for i in range(len(train_df)):
            ecg = scipy.io.loadmat(f'{path}\Train\FilteredECG\{train_df["Recording"][i]}.mat')['ECG']

            recording = [ecg, train_df['Recording'][i]]
            total_data.append(recording)

            print_progressBar(i+1, len(train_df), prefix='Filtering Train ECG:', length=50)
    
    ### Filtering Test Data
    if not os.path.exists(f'{path}\Test\FilteredECG'): os.mkdir(f'{path}\Test\FilteredECG')
    if len(os.listdir(f'{path}\Test\FilteredECG')) == 0:
        for i in range(len(test_df)):
            ecg = scipy.io.loadmat(f'{path}\Test\InitialSet\{test_df["Recording"][i]}.mat')['ECG'][0][0][2]
        
            ### Filtering EKG
            ecg = filter_ecg(ecg)[:, 100:FRAGMENT_SIZE+100]

            scipy.io.savemat(f'{path}\Test\FilteredECG\{test_df["Recording"][i]}.mat', {'ECG': ecg})

            recording = [ecg, test_df['Recording'][i]]        
            total_data.append(recording)

            print_progressBar(i+1, len(test_df), prefix='Filtering Test ECG:', length=50)
    else:
        for i in range(len(test_df)):
            ecg = scipy.io.loadmat(f'{path}\Test\FilteredECG\{test_df["Recording"][i]}.mat')['ECG']

            recording = [ecg, test_df['Recording'][i]]
            total_data.append(recording)

            print_progressBar(i+1, len(test_df), prefix='Filtering Test ECG:', length=50)

    print("Filtering done!")



    
    # channel_means, channel_stds = get_channel_means_stds(total_data[:len(train_df)])
    channel_means, channel_stds = get_channel_means_stds(total_data)

    ## Normalizing filtered data
    if not os.path.exists(f'{path}\Train\\NormFilteredECG'): os.mkdir(f'{path}\Train\\NormFilteredECG')
    if not os.path.exists(f'{path}\Test\\NormFilteredECG'): os.mkdir(f'{path}\Test\\NormFilteredECG')
    if len(os.listdir(f'{path}\Train\\NormFilteredECG')) == 0 or len(os.listdir(f'{path}\Test\\NormFilteredECG')) == 0:
        for i, recording in enumerate(total_data):
            
            for j in range(12):
                recording[0][j] = (recording[0][j] - channel_means[j]) / channel_stds[j]

            if i < len(train_df): scipy.io.savemat(f'{path}\Train\\NormFilteredECG\{recording[1]}.mat', {'ECG': recording[0]})
            else:                 scipy.io.savemat(f'{path}\Test\\NormFilteredECG\{recording[1]}.mat', {'ECG': recording[0]})

            print_progressBar(i+1, len(total_data), prefix='Normalizing filtered ECG:', length=50)

    print(f"Filtered ECG normaization done!")


    
    del total_data



    ## Normalizing initial data
    if not os.path.exists(f'{path}\Train\\NormECG'): os.mkdir(f'{path}\Train\\NormECG')
    if not os.path.exists(f'{path}\Test\\NormECG'): os.mkdir(f'{path}\Test\\NormECG')
    if len(os.listdir(f'{path}\Train\\NormECG')) == 0 or len(os.listdir(f'{path}\Test\\NormECG')) == 0:
        for i in range(len(train_df) + len(test_df)):
            
            if i < len(train_df): 
                ecg = scipy.io.loadmat(f'{path}\Train\InitialSet\{train_df["Recording"][i]}.mat')['ECG'][0][0][2][:, 100:FRAGMENT_SIZE+100]

                for j in range(12):
                    ecg[0][j] = (ecg[0][j] - channel_means[j]) / channel_stds[j]

                scipy.io.savemat(f'{path}\Train\\NormECG\{train_df["Recording"][i]}.mat', {'ECG': ecg})

            else: 
                cp_i = i % len(train_df)
                ecg = scipy.io.loadmat(f'{path}\Test\InitialSet\{test_df["Recording"][cp_i]}.mat')['ECG'][0][0][2][:, 100:FRAGMENT_SIZE+100]

                for j in range(12):
                    ecg[0][j] = (ecg[0][j] - channel_means[j]) / channel_stds[j]

                scipy.io.savemat(f'{path}\Test\\NormECG\{test_df["Recording"][cp_i]}.mat', {'ECG': ecg})

            print_progressBar(i+1, (len(train_df) + len(test_df)), prefix='Normalizing Initial ECG:', length=50)

    print(f"Initial ECG normaization done!")



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