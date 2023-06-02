import os
import pandas as pd
import scipy
from Filtering.Neurokit2Filters import filter_ecg
import numpy as np
import shutil

np.random.seed(42)
FRAGMENT_SIZE = 3000

def prepare_dataset(path='Data/ChineseDataset'):

    df = pd.read_csv(f'{path}/REFERENCE.csv', delimiter=',')

    # "Sorting" ecgs to different folders ###########################################################
    if not os.path.exists(f'{path}/Train/Initial'): 
        os.makedirs(f'{path}/Train/Initial', exist_ok=True)
        ref_train = pd.DataFrame(columns=['Recording', 'First_label', 'Second_label', 'Third_label'])
    if not os.path.exists(f'{path}/Val/Initial'):
        os.makedirs(f'{path}/Val/Initial', exist_ok=True)
        ref_val = pd.DataFrame(columns=['Recording', 'First_label', 'Second_label', 'Third_label'])
    if not os.path.exists(f'{path}/Test/Initial'):
        os.makedirs(f'{path}/Test/Initial', exist_ok=True)
        ref_test = pd.DataFrame(columns=['Recording', 'First_label', 'Second_label', 'Third_label'])

    if len(os.listdir(f'{path}/Train/Initial')) + \
        len(os.listdir(f'{path}/Val/Initial')) + \
        len(os.listdir(f'{path}/Test/Initial')) < len(os.listdir(f'{path}/InitialSet')):

        for i, ecg_file in enumerate(os.listdir(f'{path}/InitialSet')):

            row = df.loc[df['Recording'] == ecg_file.split('.')[0]]

            decision = np.random.random()
            if decision < 0.6:
                folder = 'Train'
                ref_train = pd.concat([ref_train, row], ignore_index=True)
            elif decision >= 0.6 and decision < 0.8:
                folder = 'Val'
                ref_val = pd.concat([ref_val, row], ignore_index=True)
            else:
                folder = 'Test'
                ref_test = pd.concat([ref_test, row], ignore_index=True)

            shutil.copy(src=f'{path}/InitialSet/{ecg_file}', dst=f'{path}/{folder}/Initial/{ecg_file}')

            print_progressBar(i+1, len(os.listdir(f'{path}/InitialSet')), prefix='"Sorting" ecgs to different folders', length=50)

        ref_train.to_csv(f'{path}/Train/LOCAL_REFERENCE.csv', index=False)
        ref_val.to_csv(f'{path}/Val/LOCAL_REFERENCE.csv', index=False)
        ref_test.to_csv(f'{path}/Test/LOCAL_REFERENCE.csv', index=False)
    else:
        ref_train = pd.read_csv(f'{path}/Train/LOCAL_REFERENCE.csv')
        ref_val = pd.read_csv(f'{path}/Val/LOCAL_REFERENCE.csv')
        ref_test = pd.read_csv(f'{path}/Test/LOCAL_REFERENCE.csv')

    refs = [ref_train, ref_val, ref_test]





    total_data = []
    FOLDERS = ['Train', 'Val', 'Test']

    ### Filtering Data ###########################################################################################
    for i, folder in enumerate(FOLDERS):

        local_ref = refs[i]

        if not os.path.exists(f'{path}/{folder}/FilteredECG'): os.mkdir(f'{path}/{folder}/FilteredECG')
        if len(os.listdir(f'{path}/{folder}/FilteredECG')) < len(local_ref):
            for j in range(len(local_ref)):
                ecg = scipy.io.loadmat(f'{path}/{folder}/Initial/{local_ref["Recording"][j]}.mat')['ECG'][0][0][2]
            
                ecg = filter_ecg(ecg)[:, 100:FRAGMENT_SIZE+100]

                scipy.io.savemat(f'{path}/{folder}/FilteredECG/{local_ref["Recording"][j]}.mat', {'ECG': ecg})

                # if folder == 'Train':
                recording = [ecg, local_ref['Recording'][j]]        
                total_data.append(recording)

                print_progressBar(j+1, len(local_ref), prefix=f'Filtering {folder} ECG:', length=50)
        else:
            for j in range(len(local_ref)):
                # if folder == 'Train':
                ecg = scipy.io.loadmat(f'{path}/{folder}/FilteredECG/{local_ref["Recording"][j]}.mat')['ECG']

                recording = [ecg, local_ref['Recording'][j]]
                total_data.append(recording)

                print_progressBar(j+1, len(local_ref), prefix=f'Reading filtered {folder} ECG:', length=50)

    print("Filtering done!")



    

    channel_means, channel_stds = get_channel_means_stds(total_data)
    del total_data

    ## Normalizing filtered data #################################################################################
    for i, folder in enumerate(FOLDERS):

        local_ref = refs[i]

        if not os.path.exists(f'{path}/{folder}/NormFilteredECG'): os.mkdir(f'{path}/{folder}/NormFilteredECG')
        if len(os.listdir(f'{path}/{folder}/NormFilteredECG')) < len(local_ref):
            for j in range(len(local_ref)):

                ecg = scipy.io.loadmat(f'{path}/{folder}/FilteredECG/{local_ref["Recording"][j]}.mat')['ECG']
                
                for k in range(12):
                    ecg[0][k] = (ecg[0][k] - channel_means[k]) / channel_stds[k]

                scipy.io.savemat(f'{path}/{folder}/NormFilteredECG/{local_ref["Recording"][j]}.mat', {'ECG': ecg})

                print_progressBar(j+1, len(local_ref), prefix=f'Normalizing filtered ECG in {folder}:', length=50)

    print(f"Filtered ECG normaization done!")



    ## Normalizing initial data ###########################################################################################################
    for i, folder in enumerate(FOLDERS):

        local_ref = refs[i]

        if not os.path.exists(f'{path}/{folder}/NormECG'): os.mkdir(f'{path}/{folder}/NormECG')
        if len(os.listdir(f'{path}/{folder}/NormECG')) < len(local_ref):
            for j in range(len(local_ref)):
                
                ecg = scipy.io.loadmat(f'{path}/{folder}/Initial/{local_ref["Recording"][j]}.mat')['ECG'][0][0][2][:, 100:FRAGMENT_SIZE+100]

                for k in range(12):
                    ecg[0][k] = (ecg[0][k] - channel_means[k]) / channel_stds[k]

                scipy.io.savemat(f'{path}/{folder}/NormECG/{local_ref["Recording"][j]}.mat', {'ECG': ecg})

                print_progressBar(j+1, len(local_ref), prefix=f'Normalizing Initial ECG in {folder}:', length=50)

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