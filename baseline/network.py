import torch
from torch import nn
from data import TictactoeDataset


class Neuralbaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,5), # 1 28 84 -> 1 24 80
            nn.MaxPool2d(2,2), # 6 12 40
            nn.ReLU(True),
            nn.Conv2d(6,16,5),# 16 8 36
            nn.MaxPool2d(2,2), # 16 4 18
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 18, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300,90),
            nn.ReLU(),
            nn.Linear(90,30),
            nn.ReLU(),
            nn.Linear(30, 3),
            nn.Softmax(1),
        )

    def forward(self, x):
        # x1 = self.encoder(x[0])
        # x2 = self.encoder(x[1])
        # x3 = self.encoder(x[2])

        # x1 = x1.view(-1, 16 * 4 * 4)
        # x2 = x2.view(-1, 16 * 4 * 4)
        # x3 = x3.view(-1, 16 * 4 * 4)

        # x = x1 + x2 + x3

        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 18)
        x = self.classifier(x)
        x = torch.squeeze(x)
        return x





