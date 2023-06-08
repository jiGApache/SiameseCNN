import sys
import os 
dir_path = os.path.dirname(__file__)[:os.path.dirname(__file__).rfind('\\')]
sys.path.append(dir_path)

from Datasets.Chinese.NoisyDataset import NoisyPairsDataset as DS_Noisy
# from Datasets.Physionet.NoisyDataset import PairsDataset as DS_Noisy
import numpy as np
from Models.SiameseModel import Siamese
import torch
from torch.utils.data import DataLoader 
import os
import random
import math
import json

def contrastive_loss(emb_1, emb_2, y):

    distances = torch.zeros(len(y), dtype=torch.float32)
    for i in range(len(y)):
        if y[i] == 1:
            distances[i] = torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2))
        else:
            distances[i] = torch.maximum(torch.as_tensor(0.), LOSS_MARGIN - torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2)))

    loss = torch.mean(distances)

    return loss

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

# Hyper params 
#########################################################
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 1

LR = 0.001

LOSS_FUNCTION = contrastive_loss
BATCH_SIZE = 256
WEIGHT_DECAY = 0.001
LOSS_MARGIN = 0. # Initializes in train\test loop
CLASSES = [8, 9] # Infartion labels
#########################################################

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

# train_ds = DS_Noisy(labels=CLASSES, folder='Train', samples_per_element=2)
# val_ds = DS_Noisy(labels=CLASSES, folder='Val', samples_per_element=2)
train_ds = DS_Noisy(folder='Train', samples_per_element=2)
val_ds = DS_Noisy(folder='Val', samples_per_element=2)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True, drop_last=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, persistent_workers=True, drop_last=True)

def train_epoch(epoch_counter):

    steps_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in train_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        # print(out_emb_1.shape)
        # exit()
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil((len(train_ds) - len(train_ds) % BATCH_SIZE) / BATCH_SIZE), prefix=f'{epoch_counter} Train epoch progress:', length=50)

    return epoch_loss / steps_in_epoch


def test_epoch(epoch_counter):
    
    steps_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in val_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil((len(val_ds) - len(val_ds) % BATCH_SIZE) / BATCH_SIZE), prefix=f'{epoch_counter} Test epoch progress:', length=50)

    return epoch_loss / steps_in_epoch


if __name__ == '__main__':

    if not os.path.exists('history'): os.mkdir('history')
    if not os.path.exists('nets'):    os.mkdir('nets')

    distances = [1., 4., 8., 10., 15., 20.]
    # distances = [10.]

    for distance in distances:

        history = {
            'epochs' : [],
            'train_losses' : [],
            'test_losses' : []
        }

        LOSS_MARGIN = distance
        model = Siamese().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

            print(f'Epoch: {epoch+1} Train loss: {train_loss:.5f} Validation loss:  {test_loss:.5f}\n\n')

        torch.save(model.state_dict(), f'nets\SCNN_d={distance}_labels={len(CLASSES)}.pth')

        with open(f'history\history_d={distance}_labels={len(CLASSES)}.json', 'w') as history_file:
            history_file.write(json.dumps(history))