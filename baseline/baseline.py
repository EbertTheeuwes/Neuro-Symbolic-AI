import torch
from torch import nn
from data import TictactoeDatasetCat, TictactoeDatasetSep, TictactoeDatasetSep2x3, TictactoeDatasetSep3x3
import torch.optim as optim
import torchvision
from network import NeuralbaselineCat, NeuralbaselineSep, NeuralbaselineSep2x3, NeuralbaselineSep3x3
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time


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
    iterations = []
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

            # if torch.argmax(y) == 1 == torch.argmax(norm_pred):
            #    print("found example of 5 wins correctly classified", norm_pred)

            # if torch.argmax(y) == 1:
            #     print("example where 5 wins")
            #     if torch.argmax(norm_pred) == 1:
            #         print("CORRECTLY CLASSIFIED")

            if total_iteration % 100 == 0:
                avg_loss = running_loss/amount_examined
                avg_accuracy = running_correct/amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)
                iterations.append(total_iteration)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

    return network, avg_accuracies, avg_losses, iterations

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
    iterations = []
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

            if total_iteration % 100 == 0:
                avg_loss = running_loss / amount_examined
                avg_accuracy = running_correct / amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)
                iterations.append(total_iteration)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

            softmax = torch.nn.Softmax(0)
            norm_pred = softmax(pred)

            # if torch.argmax(y) == 1 == torch.argmax(norm_pred):
            #     print("found example of 5 wins correctly classified", norm_pred)

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

def train_sep_image_model_2x3():

    trainingdata = TictactoeDatasetSep2x3("train")
    network = NeuralbaselineSep2x3()
    adamopt = optim.Adam(network.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()
    dl = DataLoader(trainingdata, batch_size=None)

    running_loss = 0
    running_correct = 0
    amount_examined = 0
    avg_losses = []
    avg_accuracies = []
    iterations = []
    total_iteration = 0
    times = []

    first_time = time.time()
    for epoch in range(0, 10):
        if epoch == 1:
            test_set = TictactoeDatasetSep2x3("test")
            accuracy, confusion_matrix = test_sep_img_network(network, test_set)
            print("accuracy after one epoch ", accuracy)

        for i, (x, y) in enumerate(dl):
            adamopt.zero_grad()
            pred = network(x[0], x[1], x[2], x[3], x[4], x[5])
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

            # if torch.argmax(y) == 1 == torch.argmax(norm_pred):
            #     print("found example of 5 wins correctly classified", norm_pred)

            # if torch.argmax(y) == 1:
            #     print("example where 5 wins")
            #     if torch.argmax(norm_pred) == 1:
            #         print("CORRECTLY CLASSIFIED")

            if total_iteration % 100 == 0:
                avg_loss = running_loss/amount_examined
                avg_accuracy = running_correct/amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)
                iterations.append(total_iteration)
                current_time = time.time()
                relative_time = current_time - first_time
                times.append(relative_time)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

    return network, avg_accuracies, avg_losses, iterations, times

def train_sep_image_model_3x3():

    trainingdata = TictactoeDatasetSep3x3("train")
    network = NeuralbaselineSep3x3()
    adamopt = optim.Adam(network.parameters(), lr=0.001)
    lossfunc = nn.CrossEntropyLoss()
    dl = DataLoader(trainingdata, batch_size=None)

    running_loss = 0
    running_correct = 0
    amount_examined = 0
    avg_losses = []
    avg_accuracies = []
    iterations = []
    total_iteration = 0

    for epoch in range(0, 20):
        for i, (x, y) in enumerate(dl):
            adamopt.zero_grad()
            pred = network(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])
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

            # if torch.argmax(y) == 1 == torch.argmax(norm_pred):
            #  print("found example of 5 wins correctly classified", norm_pred)

            # if torch.argmax(y) == 1:
            #     print("example where 5 wins")
            #     if torch.argmax(norm_pred) == 1:
            #         print("CORRECTLY CLASSIFIED")

            if total_iteration % 100 == 0:
                avg_loss = running_loss/amount_examined
                avg_accuracy = running_correct/amount_examined
                avg_losses.append(avg_loss)
                avg_accuracies.append(avg_accuracy)
                iterations.append(total_iteration)

                amount_examined = 0
                running_loss = 0
                running_correct = 0

                print("Epoch", epoch, " Iteration", i, " average loss = ", avg_loss)
                print("Epoch", epoch, " Iteration", i, " accuracy ", avg_accuracy, "%")

    return network, avg_accuracies, avg_losses, iterations


def test_sep_img_network(network, test_set):
    dl = DataLoader(test_set, batch_size=None)
                # Actual 0  5  9
    confusion_matrix = [[0, 0, 0], # Pred 0
                        [0, 0, 0], # Pred 5
                        [0, 0, 0]] # Pred 9
    correct = 0
    total = 0

    for i, (x, y) in enumerate(dl):
        pred = network(x[0], x[1], x[2], x[3], x[4], x[5]) # adapt to current grid
        row = torch.argmax(pred)
        index = torch.argmax(y)
        confusion_matrix[row][index] += 1

        if row == index:
            correct += 1
        total += 1

    accuracy = correct/total
    return accuracy, confusion_matrix


network, avg_accuracies, avg_losses, iterations, times = train_sep_image_model_2x3()
plt.plot(iterations, avg_losses)
plt.title("Loss as a function of training iterations")
plt.xlabel("training iterations")
plt.ylabel("cross entropy loss")
plt.show()

plt.plot(times, avg_losses)
plt.title("Loss as a function of time")
plt.xlabel("time")
plt.ylabel("cross entropy loss")
plt.show()


test_set = TictactoeDatasetSep2x3("test")
accuracy, confusion_matrix = test_sep_img_network(network, test_set)
print("Accuracy = ", accuracy)
print("Confusion matrix", confusion_matrix)



