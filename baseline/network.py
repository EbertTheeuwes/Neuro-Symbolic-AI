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
            # nn.Softmax(1),
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
            # nn.Softmax(1),
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

class NeuralbaselineSep2x3(nn.Module):
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
            nn.Linear(16 * 4 * 4 * 6, 450),
            nn.ReLU(),
            nn.Linear(450, 200),
            nn.ReLU(),
            nn.Linear(200, 90),
            nn.ReLU(),
            nn.Linear(90, 3),
            # nn.Softmax(1),
        )

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        x4 = self.encoder(x4)
        x5 = self.encoder(x5)
        x6 = self.encoder(x6)

        x1 = x1.view(-1, 16 * 4 * 4)
        x2 = x2.view(-1, 16 * 4 * 4)
        x3 = x3.view(-1, 16 * 4 * 4)
        x4 = x4.view(-1, 16 * 4 * 4)
        x5 = x5.view(-1, 16 * 4 * 4)
        x6 = x6.view(-1, 16 * 4 * 4)

        x1 = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        x1 = self.classifier(x1)

        return torch.squeeze(x1)

class NeuralbaselineSep3x3(nn.Module):
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
            nn.Linear(16 * 4 * 4 * 9, 500),
            nn.ReLU(),
            nn.Linear(500, 220),
            nn.ReLU(),
            nn.Linear(220, 90),
            nn.ReLU(),
            nn.Linear(90, 3),
            # nn.Softmax(1),
        )

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        x4 = self.encoder(x4)
        x5 = self.encoder(x5)
        x6 = self.encoder(x6)
        x7 = self.encoder(x7)
        x8 = self.encoder(x8)
        x9 = self.encoder(x9)

        x1 = x1.view(-1, 16 * 4 * 4)
        x2 = x2.view(-1, 16 * 4 * 4)
        x3 = x3.view(-1, 16 * 4 * 4)
        x4 = x4.view(-1, 16 * 4 * 4)
        x5 = x5.view(-1, 16 * 4 * 4)
        x6 = x6.view(-1, 16 * 4 * 4)
        x7 = x7.view(-1, 16 * 4 * 4)
        x8 = x8.view(-1, 16 * 4 * 4)
        x9 = x9.view(-1, 16 * 4 * 4)

        x1 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
        x1 = self.classifier(x1)

        return torch.squeeze(x1)

