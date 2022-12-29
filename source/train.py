from pairsDataset import PairsDataset
from model import Siamese
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt

torch.manual_seed(42)

def contrastive_loss(y_pred, y_true):
    margin = 1
    square_pred = torch.square(y_pred)
    margin_square = torch.square(torch.maximum(margin - y_pred, torch.tensor((0.))))
    return y_true * square_pred + (1 - y_true) * margin_square


def normalise_pair_ts(ts1, ts2):
    norm_ts1 = []
    norm_ts2 = []
    for tensor in ts1:
        tn_mean = torch.mean(tensor)
        tn_std = torch.std(tensor, unbiased=False)
        norm_ts1.append((tensor - tn_mean) / tn_std)
    for tensor in ts2:
        tn_mean = torch.mean(tensor)
        tn_std = torch.std(tensor, unbiased=False)
        norm_ts2.append((tensor - tn_mean) / tn_std)
    return torch.stack(norm_ts1), torch.stack(norm_ts2)


def show_history(history):
    plt.plot(history['epochs'], history['losses'], label='loss')
    plt.plot(history['epochs'], history['accuracies'], label='accuracy')
    plt.ylim([0, 1.5])
    plt.xlabel('Epoch')
    plt.ylabel('Learning')
    plt.legend()
    plt.grid(True)
    plt.show()


# Hyper params
#########################################################
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
LR = 0.001
LOSS_FUNCTION = nn.BCELoss()
BATCH_SIZE = 10
WEIGHT_DECAY = 0.001
# LOSS_FUNCTION = contrastive_loss
model = Siamese().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#########################################################


ds = PairsDataset()
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


history = {
    'epochs' : [0],
    'losses' : [],
    'accuracies' : [0]
}
correct_preds = []

for epoch in range(EPOCHS):

    steps_in_epoch = 0
    
    correct_predictions_in_epoch = 0
    running_correct_predictions = 0
    
    epoch_loss = 0.0
    running_loss = 0.0
    
    model.train()

    for TS_T, label in dl:

        steps_in_epoch += 1

        if DEVICE == 'cuda:0':  TS1, TS2, label = TS_T[0].cuda(), TS_T[1].cuda(), label.cuda()
        else:                   TS1, TS2, label = TS_T[0], TS_T[1], label

        out = model(TS1, TS2)
        out = torch.reshape(out, (-1,))
        loss = LOSS_FUNCTION(out, label)

        running_loss += loss.item()
        epoch_loss += loss.item()
        if len(history['losses']) == 0: history['losses'].append(loss.item())
        correct_predictions_in_epoch += (torch.abs(out - label) < 0.5).count_nonzero().item()
        running_correct_predictions += (torch.abs(out - label) < 0.5).count_nonzero().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps_in_epoch % 10 == 0:
            print(f'[{epoch + 1}, {steps_in_epoch:5d}] loss: {running_loss / 10:.5f} accuracy: {running_correct_predictions / (10 * BATCH_SIZE):.5f}')
            running_correct_predictions = 0
            running_loss = 0.0
    
    # print(f'[{epoch + 1}, {steps_in_epoch:5d}]  loss: {running_loss / (steps_in_epoch % 40):.10f}  accuracy: {running_correct_predictions / (steps_in_epoch * BATCH_SIZE % 40):.5f}')

    history['epochs'].append(epoch + 1)
    history['losses'].append(epoch_loss / steps_in_epoch)
    history['accuracies'].append(correct_predictions_in_epoch / (steps_in_epoch * BATCH_SIZE))
        
show_history(history)

torch.save(model.state_dict(), 'nets\\SCNN.pth')