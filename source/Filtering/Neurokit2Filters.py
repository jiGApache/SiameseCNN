import neurokit2 as nk
import numpy as np

METHOD = "neurokit"

def filter_ecg(ecg):
    filtered_ecg = []
    for i in range(12):
        filtered_ecg.append(nk.ecg_clean(ecg[i], sampling_rate=500, method=METHOD))
    filtered_ecg = np.asarray(filtered_ecg)
    return filtered_ecg