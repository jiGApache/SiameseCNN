from model import Siamese
import pandas as pd
import scipy.io
import torch
from PreprocessingFilters import filter1
from pairsDataset import PairsDataset
import matplotlib.pyplot as plt

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Siamese().to(DEVICE)
model.load_state_dict(torch.load('nets\\30SCNN.pth'))

torch.manual_seed(42)
with torch.no_grad():

    ds = PairsDataset()
    pair, label= ds.__getitem__(3200)

    if DEVICE == 'cuda:0':  in1, in2, label = pair[0][None, :, :].cuda(), pair[1][None, :, :].cuda(), label.cuda()
    else:                   in1, in2, label = pair[0][None, :, :], pair[1][None, :, :], label

    
    out = model(in1, in2)
    print('Similarity score: ', out.item())
    print('True value: ', label.item())

    # plt.plot(pair[0][0], 'r')
    # plt.show()
    