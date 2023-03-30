import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Datasets.Chinese.NoisyDataset import NoisyPairsDataset as DS_Noisy
# from Datasets.Physionet.NoisyDataset import PairsDataset as DS_Noisy
import numpy as np
from Models.SiameseModel import Siamese
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import random
import math
import json

def contrastive_loss(emb_1, emb_2, y):

    distances = []
    for i in range(len(y)):
        if y[i] == 1:
            distances.append(torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2)))    
        else:
            distances.append(torch.maximum(torch.tensor(0.), LOSS_MARGIN - torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2))))

    distances = torch.cat(distances).to(DEVICE)
    loss = torch.mean(distances)

    return loss

# Hyper params 
#########################################################
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

LR = 0.001
STEP_SIZE = EPOCHS / 5
GAMMA = 0.75

LOSS_FUNCTION = nn.BCELoss().cuda()
LOSS_FUNCTION = contrastive_loss
BATCH_SIZE = 256
WEIGHT_DECAY = 0.001
THRESHHOLD = 0.5
LOSS_MARGIN = 0. # Initializes in train\test loop
CLASSES = [8, 9]
#########################################################

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True


def show_history(history):

    plt.plot(history['epochs'], history['train_losses'], label='Train loss')
    plt.plot(history['epochs'], history['test_losses'], label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

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

full_ds = DS_Noisy(CLASSES)
train_size = int(0.8 * full_ds.__len__())
test_size = full_ds.__len__() - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True,  generator=torch.Generator().manual_seed(SEED))


def train_epoch(epoch_counter):

    steps_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in train_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil(train_size / BATCH_SIZE), prefix=f'{epoch_counter} Train epoch progress:', length=50)

    return epoch_loss / steps_in_epoch


def test_epoch(epoch_counter):
    
    steps_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in test_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil(test_size / BATCH_SIZE), prefix=f'{epoch_counter} Test epoch progress:', length=50)

    return epoch_loss / steps_in_epoch


if __name__ == '__main__':

    if not os.path.exists('history'):
        os.mkdir('history')
    if not os.path.exists('nets'):
        os.mkdir('nets')

    distances = [1., 2., 3., 4., 8., 20.]

    for distance in distances:

        history = {
            'epochs' : [],
            'train_losses' : [],
            'test_losses' : []
        }

        LOSS_MARGIN = distance
        model = Siamese().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        # scheduler = StepLR(optimizer, 
        #            step_size = STEP_SIZE,
        #            gamma = GAMMA)

        for epoch in range(EPOCHS):

            model.train(True)
            train_loss = train_epoch(epoch + 1)
            torch.cuda.empty_cache()

            model.train(False)
            test_loss = test_epoch(epoch + 1)
            torch.cuda.empty_cache()

            history['epochs'].append(epoch + 1)
            history['train_losses'].append(train_loss)
            history['test_losses'].append(test_loss)

            print(f'Epoch: {epoch+1} Train loss: {train_loss:.5f} Test loss:  {test_loss:.5f}\n\n')

            # scheduler.step()

        torch.save(model.state_dict(), f'nets\SCNN_d={distance}_labels={len(CLASSES)}.pth')

        with open(f'history\history_d={distance}_labels={len(CLASSES)}.json', 'w') as history_file:
            history_file.write(json.dumps(history))