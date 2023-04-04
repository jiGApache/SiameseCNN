import torch
import torch.nn as nn

torch.manual_seed(42)

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.slope = 0.1

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(12, 128, kernel_size=100, stride=100),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(kernel_size=2)
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(128, 64, kernel_size=5, stride=3),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(negative_slope=self.slope),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Flatten(start_dim=1)
        # )

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 128, kernel_size=400),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=self.slope),

            # nn.Dropout(0.2),

            # nn.Conv1d(128, 128, kernel_size=12, stride=10),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(negative_slope=self.slope)#,

            nn.MaxPool1d(kernel_size=10)
        )

        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=40),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=self.slope),

            # nn.Dropout(0.2),

            # nn.Conv1d(256, 256, kernel_size=12, stride=10),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(negative_slope=self.slope)#,

            nn.MaxPool1d(kernel_size=10)
        )

        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),

            nn.Conv1d(256, 16, kernel_size=4),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=self.slope),

            # nn.Dropout(0.2),

            # nn.Conv1d(16, 16, kernel_size=12, stride=10),
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(negative_slope=self.slope),

            nn.MaxPool1d(kernel_size=8),
            nn.Flatten(start_dim=1),
            nn.Tanh()
        )


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.embedding(x)
        return x

    def forward(self, x1, x2):
        # out1 = self.forward_once(x1)
        # out2 = self.forward_once(x2)
        # out = self.classifier(torch.abs(out1 - out2))
        return self.forward_once(x1), self.forward_once(x2)