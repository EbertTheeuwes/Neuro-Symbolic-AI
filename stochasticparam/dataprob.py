from typing import Mapping, Iterator
import torch
import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, Constant
from deepproblog.dataset import Dataset
from deepproblog.query import Query
import random

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

full_mnist_trainset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
full_mnist_testset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)

# Stel trainset en testset op voor 5 en 9
mnist_trainset = []
for i in range(len(full_mnist_trainset)):
    if int(full_mnist_trainset[i][1]) == 5 or int(full_mnist_trainset[i][1]) == 9:
        mnist_trainset += [full_mnist_trainset[i]]
mnist_testset = []
for i in range(len(full_mnist_testset)):
    if int(full_mnist_testset[i][1]) == 5 or int(full_mnist_testset[i][1]) == 9:
        mnist_testset += [full_mnist_testset[i]]

# Stel trainset en testset op voor 0 en 1
mnist_trainset01 = []
for i in range(len(full_mnist_trainset)):
    if int(full_mnist_trainset[i][1]) == 0 or int(full_mnist_trainset[i][1]) == 1:
        mnist_trainset01 += [full_mnist_trainset[i]]
mnist_testset01 = []
for i in range(len(full_mnist_testset)):
    if int(full_mnist_testset[i][1]) == 0 or int(full_mnist_testset[i][1]) == 1:
        mnist_testset01 += [full_mnist_testset[i]]

#random.shuffle(mnist_trainset01)
#random.shuffle(mnist_testset01)
datasets = {
    #"train": mnist_trainset,
    #"test": mnist_testset,
    "train": mnist_trainset01,
    "test": mnist_testset01,
}

numbers = [0, 1] #[5,9]
neutral_number = 2#0

class MNISTImages(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator:
        for i in range(self.dataset):
            yield self.dataset[i][0]

    def __len__(self) -> int:
        return len(self.dataset)

    def __init__(self, subset):
        self.subset = subset
        self.dataset = datasets[self.subset]

    def __getitem__(self, item):
        return self.dataset[int(item[0])][0]



probability0 = 0.8
class AdditionDataset(Dataset):

    def __init__(self, subset):
        self.subset = subset
        self.dataset = datasets[subset]

    def __len__(self):
        return len(self.dataset) // 2

    def query1x3(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 2)))
        label1 = self.dataset[2*i][1]
        image2 = Term("tensor", Term(self.subset, Constant(i * 2 + 1)))
        label2 = self.dataset[i*2+1][1]

        # dus gekozen om telkens query te maken die zegt dat label1 wint, en probability hiervan mee te geven
        prob_win_label1 = get_prob_win_label1(label1, label2)
        term = Term('win', image1, image2, Constant(label1))
        return Query(term, p=prob_win_label1)

    def to_query(self, i: int) -> Query:
        return self.query1x3(i)


def checkgrid1(grid):
    first = grid[0]
    for i in range(len(grid)):
        if int(first) != int(grid[i]):
            return 'No'
    return first

def get_prob_win_label1(label1, label2):
    if label1 != label2:
        return 0
    else:
        if label1 == label2 == 0:
            return probability0
        if label1 == label2 == 1:
            return 1 - probability0


# trainset = AdditionDataset("train")
#
# for i in range(10):
#     print(trainset.to_query(i))