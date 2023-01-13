import torch
import torch.nn as nn

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.slope = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=30, stride=15),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(5, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(24, 64, kernel_size=20, stride=10),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=16, stride=8),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=self.slope),
            nn.MaxPool1d(3, stride=3)
        )

        # self.conv4 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 256, kernel_size=12, stride=6),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(2, stride=2)
        # )

        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(256, 32, kernel_size=8, stride=4),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(2, stride=2)
        # )

        # self.conv6 = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Conv1d(32, 24, kernel_size=4, stride=2),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(2, stride=2)
        # )

        self.dense = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(66 * 24, 128)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dense(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        out = self.classifier(torch.abs(out1 - out2))
        return out