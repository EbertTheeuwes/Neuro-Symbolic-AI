import torch
from torch import nn
from data import TictactoeDatasetCat, TictactoeDatasetSep


class NeuralbaselineCat(nn.Module):
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
            nn.Linear(16 * 4 * 18, 750),
            nn.ReLU(),
            nn.Linear(750, 500),
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
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 18)
        x = self.classifier(x)
        x = torch.squeeze(x)
        return x

class NeuralbaselineSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
            nn.Conv2d(6,16,5),
            nn.MaxPool2d(2,2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4 * 3, 300),
            nn.ReLU(),
            nn.Linear(300, 140),
            nn.ReLU(),
            nn.Linear(140, 90),
            nn.ReLU(),
            nn.Linear(90, 3),
            nn.Softmax(1),
        )

    def forward(self, x, y, z):
        x = self.encoder(x)
        y = self.encoder(y)
        z = self.encoder(z)

        x = x.view(-1, 16 * 4 * 4)
        y = y.view(-1, 16 * 4 * 4)
        z = z.view(-1, 16 * 4 * 4)

        x = torch.cat((x, y, z), 1)
        x = self.classifier(x)

        return torch.squeeze(x)





