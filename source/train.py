from pairsDataset import PairsDataset
import numpy as np
from model import Siamese
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
import matplotlib.pyplot as plt

# Hyper params
#########################################################
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
LR = 0.001
# LOSS_FUNCTION = nn.BCEWithLogitsLoss().cuda()
LOSS_FUNCTION = nn.BCELoss().cuda()
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
model = Siamese().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# scaler = torch.cuda.amp.GradScaler()
#########################################################

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


def show_history(history):
    plt.plot(history['epochs'], history['train_losses'], label='Train loss')
    plt.plot(history['epochs'], history['train_accuracies'], label='Train accuracy')
    plt.plot(history['epochs'], history['test_losses'], label='Test loss')
    plt.plot(history['epochs'], history['test_accuracies'], label='Test accuracy')
    plt.ylim([0, 1.0])
    plt.xlabel('Epoch')
    plt.ylabel('Learning')
    plt.legend()
    plt.grid(True)
    plt.show()


full_ds = PairsDataset(device=DEVICE, fill_with_type='cyclic_repeat')
train_size = int(0.8 * len(full_ds))
test_size = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


history = {
    'epochs' : [0],
    'train_losses' : [1.],
    'test_losses' : [1.],
    'train_accuracies' : [0],
    'test_accuracies' : [0]
}
correct_preds = []


def train_epoch():

    steps_in_epoch = 0
    correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in train_dl:

        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE), TS_T[1].to(DEVICE), label.to(DEVICE)

        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = torch.reshape(model(TS1, TS2), (-1,))
        loss = LOSS_FUNCTION(out, label)

        epoch_loss += loss.item()
        correct_predictions_in_epoch += (torch.abs(out - label) < 0.5).count_nonzero().item()

        loss.backward()
        # scaler.scale(loss).backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad(set_to_none=True)

        del out, loss

    return correct_predictions_in_epoch / train_size, epoch_loss / steps_in_epoch


def test_epoch():
    
    steps_in_epoch = 0
    correct_predictions_in_epoch = 0
    epoch_loss = 0.0

    for TS_T, label in test_dl:

        steps_in_epoch += 1

        TS1, TS2, label = TS_T[0].to(DEVICE, non_blocking=True), TS_T[1].to(DEVICE, non_blocking=True), label.to(DEVICE, non_blocking=True)

        out = torch.reshape(model(TS1, TS2), (-1,))
        loss = LOSS_FUNCTION(out, label)

        epoch_loss += loss.item()
        correct_predictions_in_epoch += (torch.abs(out - label) < 0.5).count_nonzero().item()

        del out, loss

    return correct_predictions_in_epoch / test_size, epoch_loss / steps_in_epoch


if __name__ == '__main__':

    for epoch in range(EPOCHS):

        model.train(True)
        train_acc, train_loss = train_epoch()
        torch.cuda.empty_cache()

        model.train(False)
        test_acc, test_loss = test_epoch()
        torch.cuda.empty_cache()

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracies'].append(train_acc)
        history['test_accuracies'].append(test_acc)

        print(f'Epoch: {epoch+1}\n\tTrain accuracy: {train_acc:.5f} -- Train loss: {train_loss:.5f}\n\tTest accuracy:  {test_acc:.5f} -- Test loss:  {test_loss:.5f}\n\n')
            
    torch.save(model.state_dict(), 'nets\\SCNN.pth')

    show_history(history)