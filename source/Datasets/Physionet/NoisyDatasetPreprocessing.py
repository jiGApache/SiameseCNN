import wfdb
import ast
import pandas as pd
import numpy as np
import os
import scipy.io
from Filtering.Neurokit2Filters import filter_ecg
# from Filtering.PreprocessingFilters import filter_ecg


np.random.seed(42)

FRAGMENT_SIZE = 3000
sampling_rate=500



def load_raw_data(df, sampling_rate, path): # Reading ECG signals
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic, agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))



def prepare_dataset(path='Data/PTB-XL/'):
    

    DATA_TYPES = ['Train', 'Val', 'Test']
    

    # "Sorting" ecgs to different folders ###########################################################
    if not os.path.exists(f'{path}/Train'): os.makedirs(f'{path}/Train/Initial')
    if not os.path.exists(f'{path}/Val'): os.makedirs(f'{path}/Val/Initial')
    if not os.path.exists(f'{path}/Test'): os.makedirs(f'{path}/Test/Initial')
    if len(os.listdir(f'{path}/Train/Initial')) == 0 or \
          len(os.listdir(f'{path}/Val/Initial')) == 0 or \
              len(os.listdir(f'{path}/Test/Initial')) == 0: 

        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        # String scp_codes to dict
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y, sampling_rate, path)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic, args=(agg_df,))


        X_data = []; Y_data = []
        # Split data into train and test
        val_fold = 9
        test_fold = 10
        X_data.append(X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != val_fold))])                                # Train
        Y_data.append(pd.DataFrame(Y[((Y.strat_fold != test_fold) & (Y.strat_fold != val_fold))].diagnostic_superclass))
        X_data.append(X[np.where(Y.strat_fold == val_fold)])                                                                # Val
        Y_data.append(pd.DataFrame(Y[Y.strat_fold == val_fold].diagnostic_superclass))
        X_data.append(X[np.where(Y.strat_fold == test_fold)])                                                               # Test
        Y_data.append(pd.DataFrame(Y[Y.strat_fold == test_fold].diagnostic_superclass))

        for i, dtype in enumerate(DATA_TYPES):
            Y_data[i]['STTC'] = Y_data[i]['diagnostic_superclass'].apply(lambda x: 1 if 'STTC' in x else 0)
            Y_data[i]['NORM'] = Y_data[i]['diagnostic_superclass'].apply(lambda x: 1 if 'NORM' in x else 0)
            Y_data[i]['MI']   = Y_data[i]['diagnostic_superclass'].apply(lambda x: 1 if 'MI' in x else 0)
            Y_data[i]['HYP']  = Y_data[i]['diagnostic_superclass'].apply(lambda x: 1 if 'HYP' in x else 0)
            Y_data[i]['CD']   = Y_data[i]['diagnostic_superclass'].apply(lambda x: 1 if 'CD' in x else 0)
            Y_data[i] = Y_data[i].drop(['diagnostic_superclass'], axis=1)
            Y_data[i].to_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv')
            for i, ecg in zip(Y_data[i].index, X_data[i]):
                scipy.io.savemat(f'{path}/{dtype}/Initial/{str(i).zfill(5)}.mat', {'ECG': np.transpose(ecg)})

        del X;      del Y
        del X_data; del Y_data



    total_data = []

    ### Filtering Data ###########################################################################################
    for i, dtype in enumerate(DATA_TYPES):

        local_ref = pd.read_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv', index_col=None)

        if not os.path.exists(f'{path}/{dtype}/FilteredECG'): os.mkdir(f'{path}/{dtype}/FilteredECG')
        if len(os.listdir(f'{path}/{dtype}/FilteredECG')) < len(local_ref):
            for j in range(len(local_ref)):
                ecg = scipy.io.loadmat(f'{path}/{dtype}/Initial/{str(local_ref["ecg_id"][j]).zfill(5)}.mat')['ECG']
            
                ecg = filter_ecg(ecg)[:, 100:FRAGMENT_SIZE+100]

                scipy.io.savemat(f'{path}/{dtype}/FilteredECG/{str(local_ref["ecg_id"][j]).zfill(5)}.mat', {'ECG': ecg})

                # if folder == 'Train':
                recording = [ecg, str(local_ref['ecg_id'][j]).zfill(5)]        
                total_data.append(recording)

                print_progressBar(j+1, len(local_ref), prefix=f'Filtering {dtype} ECG:', length=50)
        else:
            for j in range(len(local_ref)):
                # if folder == 'Train':
                ecg = scipy.io.loadmat(f'{path}/{dtype}/FilteredECG/{str(local_ref["ecg_id"][j]).zfill(5)}.mat')['ECG']

                recording = [ecg, str(local_ref['ecg_id'][j]).zfill(5)]
                total_data.append(recording)

                print_progressBar(j+1, len(local_ref), prefix=f'Reading filtered {dtype} ECG:', length=50)

    print("Filtering done!")




    channel_means, channel_stds = get_channel_means_stds(total_data)
    del total_data

    ## Normalizing filtered data #################################################################################
    for i, dtype in enumerate(DATA_TYPES):

        local_ref = pd.read_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv')

        if not os.path.exists(f'{path}/{dtype}/NormFilteredECG'): os.mkdir(f'{path}/{dtype}/NormFilteredECG')
        if len(os.listdir(f'{path}/{dtype}/NormFilteredECG')) < len(local_ref):
            for j in range(len(local_ref)):

                ecg = scipy.io.loadmat(f'{path}/{dtype}/FilteredECG/{str(local_ref["ecg_id"][j]).zfill(5)}.mat')['ECG']
                
                for k in range(12):
                    ecg[0][k] = (ecg[0][k] - channel_means[k]) / channel_stds[k]

                scipy.io.savemat(f'{path}/{dtype}/NormFilteredECG/{str(local_ref["ecg_id"][j]).zfill(5)}.mat', {'ECG': ecg})

                print_progressBar(j+1, len(local_ref), prefix=f'Normalizing filtered ECG in {dtype}:', length=50)

    print(f"Filtered ECG normaization done!")




    ## Normalizing initial data ###########################################################################################################
    for i, dtype in enumerate(DATA_TYPES):

        local_ref = pd.read_csv(f'{path}/{dtype}/LOCAL_REFERENCE.csv')

        if not os.path.exists(f'{path}/{dtype}/NormECG'): os.mkdir(f'{path}/{dtype}/NormECG')
        if len(os.listdir(f'{path}/{dtype}/NormECG')) < len(local_ref):
            for j in range(len(local_ref)):
                
                ecg = scipy.io.loadmat(f'{path}/{dtype}/Initial/{str(local_ref["ecg_id"][j]).zfill(5)}.mat')['ECG'][:, 100:FRAGMENT_SIZE+100]
                ecg = np.transpose(ecg)

                for k in range(12):
                    ecg[0][k] = (ecg[0][k] - channel_means[k]) / channel_stds[k]

                scipy.io.savemat(f'{path}/{dtype}/NormECG/{str(local_ref["ecg_id"][j]).zfill(5)}.mat', {'ECG': ecg})

                print_progressBar(j+1, len(local_ref), prefix=f'Normalizing Initial ECG in {dtype}:', length=50)

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