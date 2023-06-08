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
            nn.Conv1d(256, 512, kernel_size=4),
            nn.BatchNorm1d(512),
            nn.SELU(),
            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=1024, out_features=16),
            nn.Tanh()
        )


        # self.conv0 = nn.Sequential(
        #     nn.Conv1d(12, 64, kernel_size=16),
        #     nn.BatchNorm1d(64),
        #     nn.SELU()
        #     )


        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=16, padding=7, stride=2),
        #     nn.BatchNorm1d(64),
        #     nn.SELU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(64, 64, kernel_size=16, padding=7, stride=2)
        # )
        # self.res1 = nn.Sequential(
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(64, 64, kernel_size=1)
        # )


        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(64, 128, kernel_size=16, padding=7, stride=2),
        #     nn.BatchNorm1d(128),
        #     nn.SELU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 128, kernel_size=16, padding=7, stride=2)
        # )
        # self.res2 = nn.Sequential(
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(64, 128, kernel_size=1)
        # )


        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(128, 128, kernel_size=16, padding=7, stride=2),
        #     nn.BatchNorm1d(128),
        #     nn.SELU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(128, 128, kernel_size=16, padding=7, stride=2)
        # )
        # self.res3 = nn.Sequential(
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(128, 128, kernel_size=1)
        # )


        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(128, 192, kernel_size=16, padding=7, stride=2),
        #     nn.BatchNorm1d(192),
        #     nn.SELU(),
        #     nn.Dropout(0.2),
        #     nn.Conv1d(192, 192, kernel_size=16, padding=7, stride=2)
        # )
        # self.res4 = nn.Sequential(
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(128, 192, kernel_size=1)
        # )


        # self.flatten = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Tanh()
        # )


    def forward_once(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # x = self.conv0(x)
        # x = self.conv1(x) + self.res1(x)
        # x = self.conv2(x) + self.res2(x)
        # x = self.conv3(x) + self.res3(x)
        # x = self.conv4(x) + self.res4(x)
        # x = self.flatten(x)

        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)