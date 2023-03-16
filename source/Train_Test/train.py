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
            # distances.append((LOSS_MARGIN**2) / torch.square(torch.cdist(emb_1[None, i], emb_2[None, i], p=2)))

    distances = torch.cat(distances).to(DEVICE)
    loss = torch.mean(distances)

    return loss

# Hyper params 
#########################################################
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 150
LR = 0.001
LOSS_FUNCTION = nn.BCELoss().cuda()
LOSS_FUNCTION = contrastive_loss
BATCH_SIZE = 128
WEIGHT_DECAY = 0.001
THRESHHOLD = 0.5
LOSS_MARGIN = 0. # Initializes in train\test loop
CLASSES = [1, 3, 5, 7, 9]
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


# full_ds = DS_Chinese(device=DEVICE, fill_with_type='mean')
# full_ds = DS_5000()
full_ds = DS_Noisy(CLASSES)
train_size = int(0.8 * full_ds.__len__())
test_size = full_ds.__len__() - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True, generator=torch.Generator().manual_seed(SEED))
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True,  generator=torch.Generator().manual_seed(SEED))


history = {
    'epochs' : [],
    'train_losses' : [],
    'test_losses' : []#,
    # 'train_accuracies' : [],
    # 'test_accuracies' : []
}
correct_preds = []


def train_epoch(epoch_counter):

    steps_in_epoch = 0
    # correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in train_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()
        # correct_predictions_in_epoch += (torch.abs(loss - out) < THRESHHOLD).count_nonzero().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil(train_size / BATCH_SIZE), prefix=f'{epoch_counter} Train epoch progress:', length=50)

    return epoch_loss / steps_in_epoch #correct_predictions_in_epoch / train_size, epoch_loss / steps_in_epoch


def test_epoch(epoch_counter):
    
    steps_in_epoch = 0
    # correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in test_dl:
        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)
        out_emb_1, out_emb_2 = model(TS1, TS2)
        loss = LOSS_FUNCTION(out_emb_1, out_emb_2, label)

        epoch_loss += loss.item()
        # correct_predictions_in_epoch += (torch.abs(out - label) < THRESHHOLD).count_nonzero().item()

        del out_emb_1, out_emb_2, loss

        print_progressBar(steps_in_epoch, math.ceil(test_size / BATCH_SIZE), prefix=f'{epoch_counter} Test epoch progress:', length=50)

    return epoch_loss / steps_in_epoch #correct_predictions_in_epoch / test_size, epoch_loss / steps_in_epoch


if __name__ == '__main__':

    if not os.path.exists('history'):
        os.mkdir('history')
    if not os.path.exists('nets'):
            os.mkdir('nets')

    distances = [0.5, 1., 2., 3., 4., 5.]

    for distance in distances:

        LOSS_MARGIN = distance
        model = Siamese().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):

            model.train(True)
            # train_acc, train_loss = train_epoch(epoch_counter)
            train_loss = train_epoch(epoch + 1)
            torch.cuda.empty_cache()

            model.train(False)
            # test_acc, test_loss = test_epoch(epoch_counter)
            test_loss = test_epoch(epoch + 1)
            torch.cuda.empty_cache()

            history['epochs'].append(epoch + 1)
            history['train_losses'].append(train_loss)
            history['test_losses'].append(test_loss)
            # history['train_accuracies'].append(train_acc)
            # history['test_accuracies'].append(test_acc)

            # print(f'Epoch: {epoch+1}\n\tTrain accuracy: {train_acc:.5f} -- Train loss: {train_loss:.5f}\n\tTest accuracy:  {test_acc:.5f} -- Test loss:  {test_loss:.5f}\n\n')
            print(f'Epoch: {epoch+1}\tTrain loss: {train_loss:.5f}\tTest loss:  {test_loss:.5f}\n\n')

        torch.save(model.state_dict(), f'nets\SCNN_d={distance}_labels={len(CLASSES)}.pth')

        with open(f'history\history_d={distance}_labels={len(CLASSES)}.json', 'w') as history_file:
            history_file.write(json.dumps(history))