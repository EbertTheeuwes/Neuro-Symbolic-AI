import torch
from torch import nn
from data import TictactoeDatasetCat, TictactoeDatasetSep
import torch.optim as optim
import torchvision
from network import NeuralbaselineCat, NeuralbaselineSep
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



def train_sep_image_model():

    trainingdata = TictactoeDatasetSep("train")
    network = NeuralbaselineSep()
    adamopt = optim.Adam(network.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()
    dl = DataLoader(trainingdata, batch_size=None)

    running_loss = 0
    running_correct = 0
    amount_examined = 0
    avg_losses = []
    avg_accuracies = []
    total_iteration = 0

    for epoch in range(0, 10):
        for i, (x, y) in enumerate(dl):
            adamopt.zero_grad()
            pred = network(x[0], x[1], x[2])

            loss = lossfunc(pred, y)
            loss.backward()
            adamopt.step()
            running_loss += loss.item()
            amount_examined += 1
            total_iteration += 1

            if torch.argmax(pred) == torch.argmax(y):
                running_correct += 1

            softmax = torch.nn.Softmax(0)
            norm_pred = softmax(pred)

            if torch.argmax(y) == 1 == torch.argmax(norm_pred):
                print("found example of 5 wins correctly classified", norm_pred)

            # if torch.argmax(y) == 1:
            #     print("example where 5 wins")
            #     if torch.argmax(norm_pred) == 1:
            #         print("CORRECTLY CLASSIFIED")

            if total_iteration % 500 == 0:
                avg_loss = running_loss/amount_examined
                avg_accuracy = running_correct/amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

    return network, avg_accuracies, avg_losses

def train_cat_image_model():
    trainingdata = TictactoeDatasetCat("train")
    network = NeuralbaselineCat()
    adamopt = optim.Adam(network.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()
    dl = DataLoader(trainingdata, batch_size=None)

    running_loss = 0
    running_correct = 0
    amount_examined = 0
    avg_losses = []
    avg_accuracies = []
    total_iteration = 0

    for epoch in range(0, 10):
        for i, (x, y) in enumerate(dl):
            adamopt.zero_grad()
            pred = network(x)

            loss = lossfunc(pred, y)
            loss.backward()
            adamopt.step()

            running_loss += loss.item()
            amount_examined += 1
            total_iteration += 1

            if torch.argmax(pred) == torch.argmax(y):
                running_correct += 1

            if total_iteration % 500 == 0:
                avg_loss = running_loss / amount_examined
                avg_accuracy = running_correct / amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

            softmax = torch.nn.Softmax(0)
            norm_pred = softmax(pred)

            if torch.argmax(y) == 1 == torch.argmax(norm_pred):
                print("found example of 5 wins correctly classified", norm_pred)

    return network, avg_accuracies, avg_losses

# code to check if parameters actually change
# if i == 150:
#     if epoch == 0:
#         prevx = x
#     else:
#         print("Same tensor as last epoch? ", torch.equal(prevx, x))
#
#     print(x.shape)
#     # plt.imshow(x[0].squeeze())
#     # plt.title("tensor 150")
#     # plt.show()
#     print("prediction for tensor 150", pred)



train_sep_image_model()