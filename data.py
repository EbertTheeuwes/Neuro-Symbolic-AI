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


class AdditionDataset(Dataset):

    def __init__(self, subset):
        self.subset = subset
        self.dataset = datasets[subset]

    def __len__(self):
        return len(self.dataset) // 3

    def query1x3(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 3)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 3 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 3 + 2)))
        result = checkgrid1([int(self.dataset[i*3][1]), int(self.dataset[i*3 + 1][1]), int(self.dataset[i*3 + 2][1])])
        if result == 'No':
           label = Constant(int(neutral_number))
        else:
           label = Constant(int(result))
        term = Term('check1x3grid', image1, image2, image3, label)
        return Query(term)

    def query2x3(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 6)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 6 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 6 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 6 + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i * 6 + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i * 6 + 5)))
        result = checkgrid2(
           [int(self.dataset[i * 6][1]), int(self.dataset[i * 6 + 1][1]), int(self.dataset[i * 6 + 2][1])],
           [int(self.dataset[i * 6 + 3][1]), int(self.dataset[i * 6 + 4][1]), int(self.dataset[i * 6 + 5][1])])
        if result == 'No':
           label = Constant(int(neutral_number))
        else:
           label = Constant(result)
        term = Term('check2x3grid', image1, image2, image3, image4, image5, image6, label)
        return Query(term)

    def query2x2(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 4)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 4 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 4 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 4 + 3)))
        result = checkgrid22(
           [int(self.dataset[i * 4][1]), int(self.dataset[i * 4 + 1][1])],
           [int(self.dataset[i * 4 + 2][1]), int(self.dataset[i * 4 + 3][1])])
        if result == 'No':
           label = Constant(int(neutral_number))
        else:
           label = Constant(int(result))
        term = Term('check2x2grid', image1, image2, image3, image4, label)
        return Query(term)

    def query3x3(self, i: int):
        #image1 = Term("tensor", Term(self.subset, Constant(i * 9)))
        #image2 = Term("tensor", Term(self.subset, Constant(i * 9 + 1)))
        #image3 = Term("tensor", Term(self.subset, Constant(i * 9 + 2)))
        #image4 = Term("tensor", Term(self.subset, Constant(i * 9 + 3)))
        #image5 = Term("tensor", Term(self.subset, Constant(i * 9 + 4)))
        #image6 = Term("tensor", Term(self.subset, Constant(i * 9 + 5)))
        #image7 = Term("tensor", Term(self.subset, Constant(i * 9 + 6)))
        #image8 = Term("tensor", Term(self.subset, Constant(i * 9 + 7)))
        #image9 = Term("tensor", Term(self.subset, Constant(i * 9 + 8)))
        image1 = Term("tensor", Term(self.subset, Constant(i)))
        image2 = Term("tensor", Term(self.subset, Constant(i + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i + 5)))
        image7 = Term("tensor", Term(self.subset, Constant(i + 6)))
        image8 = Term("tensor", Term(self.subset, Constant(i + 7)))
        image9 = Term("tensor", Term(self.subset, Constant(i + 8)))
        #result = checkgrid3(
        #   [int(self.dataset[i * 9][1]), int(self.dataset[i * 9 + 1][1]), int(self.dataset[i * 9 + 2][1])],
        #   [int(self.dataset[i * 9 + 3][1]), int(self.dataset[i * 9 + 4][1]), int(self.dataset[i * 9 + 5][1])],
        #   [int(self.dataset[i * 9 + 6][1]), int(self.dataset[i * 9 + 7][1]), int(self.dataset[i * 9 + 8][1])])
        result = checkgrid3(
           [int(self.dataset[i][1]), int(self.dataset[i + 1][1]), int(self.dataset[i + 2][1])],
           [int(self.dataset[i + 3][1]), int(self.dataset[i + 4][1]), int(self.dataset[i + 5][1])],
           [int(self.dataset[i + 6][1]), int(self.dataset[i + 7][1]), int(self.dataset[i + 8][1])])
        if result == 'No':
           label = Constant(int(neutral_number))
        else:
           label = Constant(result)
        term = Term('check3x3grid', image1, image2, image3, image4, image5, image6, image7, image8, image9, label)
        return Query(term)

    def to_query(self, i: int) -> Query:
        return self.query1x3(i)
        #return self.query2x3(i)
        #return self.query2x2(i)
        #return self.query3x3(i)

# grid is van vorm: [A1,A2,A3]
def checkgrid1(grid):
    first = grid[0]
    for i in range(len(grid)):
        if int(first) != int(grid[i]):
            return 'No'
    return first

def checkgrid2(row1, row2):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    if numbers[0] in results:
        return numbers[0]
    elif numbers[1] in results:
        return numbers[1]
    else:
        return 'No'

def checkgrid3(row1, row2, row3):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    results += [checkgrid1(row3)]
    results += [checkgrid1([row1[0], row2[0], row3[0]])]
    results += [checkgrid1([row1[1], row2[1], row3[1]])]
    results += [checkgrid1([row1[2], row2[2], row3[2]])]
    results += [checkgrid1([row1[0], row2[1], row3[2]])]
    results += [checkgrid1([row1[2], row2[1], row3[0]])]
    if numbers[0] in results:
        return numbers[0]
    elif numbers[1] in results:
        return numbers[1]
    else:
        return 'No'

def checkgrid22(row1, row2):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    results += [checkgrid1([row1[0], row2[0]])]
    results += [checkgrid1([row1[1], row2[1]])]
    results += [checkgrid1([row1[0], row2[1]])]
    results += [checkgrid1([row1[1], row2[0]])]
    if numbers[0] in results:
        return numbers[0]
    elif numbers[1] in results:
        return numbers[1]
    else:
        return 'No'
