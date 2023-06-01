import torch
import torch.nn as nn

torch.manual_seed(42)

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.slope = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 128, kernel_size=400),
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=10)
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=40),
            nn.BatchNorm1d(256),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=10)
        )

        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(256, 16, kernel_size=4),
            nn.BatchNorm1d(16),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(start_dim=1),
            nn.Tanh()
        )


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)