import torch
from torch import nn
from data import TictactoeDataset
import torch.optim as optim
import torchvision
from network import Neuralbaseline
from torch.utils.data import Dataset, DataLoader

trainingdata = TictactoeDataset("train")
network = Neuralbaseline()
adamopt = optim.Adam(network.parameters(), lr=0.00001, weight_decay=1e-2)
lossfunc = nn.CrossEntropyLoss()
dl = DataLoader(trainingdata, batch_size=5)
avg_loss = 0

test = network(trainingdata.__getitem__(0)[0])
print(test)
correct = 0


for epoch in range(0,10):
    for i, (x, y) in enumerate(dl):
        adamopt.zero_grad()
        pred = network(x)
        if i == 1:
            print(pred)
        loss = lossfunc(pred, y)
        loss.backward()
        adamopt.step()
        avg_loss += loss.item()
        for index in range(len(pred)):
            if torch.argmax(pred[index]) == torch.argmax(y[index]):
                correct += 1

        if torch.argmax(pred) == torch.argmax(y):
            correct += 1
        if i % 500 == 0:
            avg_loss = avg_loss / 500
            print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
            print("Epoch", epoch, " Iteration", i, " accuracy ", correct/500 * 100, "%")
            avg_loss = 0
            correct = 0


