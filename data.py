from typing import Mapping, Iterator
import torch
import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, Constant, Var
from deepproblog.dataset import Dataset
from deepproblog.query import Query
import random

numbers = [5, 9]#[0, 1]
#numbers = [1, 8]
neutral_number = 0#2
grid_to_query = "1x3"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

full_mnist_trainset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
full_mnist_testset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)

# Get trainset with input numbers looking like [5,9,0]
def get_trainset(numbers, neutral):
    mnist_trainset = []
    for i in range(len(full_mnist_trainset)):
        if int(full_mnist_trainset[i][1]) in numbers:
            mnist_trainset += [full_mnist_trainset[i]]
    #for i in range(len(full_mnist_trainset)):
    #    if int(full_mnist_trainset[i][1]) in (numbers + [neutral]):
    #        mnist_trainset += [full_mnist_trainset[i]]
    return mnist_trainset

# Get testset with input numbers looking like [5,9,0]
def get_testset(numbers, neutral):
    mnist_testset = []
    for i in range(len(full_mnist_testset)):
        if int(full_mnist_testset[i][1]) in numbers:
            mnist_testset += [full_mnist_testset[i]]
    #for i in range(len(full_mnist_testset)):
    #    if int(full_mnist_testset[i][1]) in (numbers + [neutral]):
    #        mnist_testset += [full_mnist_testset[i]]
    return mnist_testset


datasets = {
    "train": get_trainset(numbers, neutral_number),
    "test": get_testset(numbers, neutral_number),
}


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
        #image1 = Term("tensor", Term(self.subset, Constant(i * 3)))
        #image2 = Term("tensor", Term(self.subset, Constant(i * 3 + 1)))
        #image3 = Term("tensor", Term(self.subset, Constant(i * 3 + 2)))
        #label = Constant(checkgrid1(
        #    [int(self.dataset[i * 3][1]), int(self.dataset[i * 3 + 1][1]), int(self.dataset[i * 3 + 2][1])]))
        #term = Term('check1x3grid', image1, image2, image3, label)
        image1 = Term("tensor", Term(self.subset, Constant(i*3)))
        image2 = Term("tensor", Term(self.subset, Constant(i*3 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i*3 + 2)))
        label = Constant(checkgrid1(
            [int(self.dataset[i*3][1]), int(self.dataset[i*3 + 1][1]), int(self.dataset[i*3 + 2][1])]))
        term = Term('check1x3grid', image1, image2, image3, label)
        return Query(term)

    def query2x3(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 6)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 6 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 6 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 6 + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i * 6 + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i * 6 + 5)))
        label = Constant(checkgrid2(
            [int(self.dataset[i * 6][1]), int(self.dataset[i * 6 + 1][1]), int(self.dataset[i * 6 + 2][1])],
            [int(self.dataset[i * 6 + 3][1]), int(self.dataset[i * 6 + 4][1]), int(self.dataset[i * 6 + 5][1])]))
        term = Term('check2x3grid', image1, image2, image3, image4, image5, image6, label)
        return Query(term)

    def query2x2(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 4)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 4 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 4 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 4 + 3)))
        label = Constant(checkgrid22(
            [int(self.dataset[i * 4][1]), int(self.dataset[i * 4 + 1][1])],
            [int(self.dataset[i * 4 + 2][1]), int(self.dataset[i * 4 + 3][1])]))
        term = Term('check2x2grid', image1, image2, image3, image4, label)
        return Query(term)

    def query3x3(self, i: int):
        # image1 = Term("tensor", Term(self.subset, Constant(i * 9)))
        # image2 = Term("tensor", Term(self.subset, Constant(i * 9 + 1)))
        # image3 = Term("tensor", Term(self.subset, Constant(i * 9 + 2)))
        # image4 = Term("tensor", Term(self.subset, Constant(i * 9 + 3)))
        # image5 = Term("tensor", Term(self.subset, Constant(i * 9 + 4)))
        # image6 = Term("tensor", Term(self.subset, Constant(i * 9 + 5)))
        # image7 = Term("tensor", Term(self.subset, Constant(i * 9 + 6)))
        # image8 = Term("tensor", Term(self.subset, Constant(i * 9 + 7)))
        # image9 = Term("tensor", Term(self.subset, Constant(i * 9 + 8)))
        image1 = Term("tensor", Term(self.subset, Constant(i)))
        image2 = Term("tensor", Term(self.subset, Constant(i + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i + 5)))
        image7 = Term("tensor", Term(self.subset, Constant(i + 6)))
        image8 = Term("tensor", Term(self.subset, Constant(i + 7)))
        image9 = Term("tensor", Term(self.subset, Constant(i + 8)))
        # label = Constant(checkgrid3(
        #   [int(self.dataset[i * 9][1]), int(self.dataset[i * 9 + 1][1]), int(self.dataset[i * 9 + 2][1])],
        #   [int(self.dataset[i * 9 + 3][1]), int(self.dataset[i * 9 + 4][1]), int(self.dataset[i * 9 + 5][1])],
        #   [int(self.dataset[i * 9 + 6][1]), int(self.dataset[i * 9 + 7][1]), int(self.dataset[i * 9 + 8][1])]))
        label = Constant(checkgrid3(
            [int(self.dataset[i][1]), int(self.dataset[i + 1][1]), int(self.dataset[i + 2][1])],
            [int(self.dataset[i + 3][1]), int(self.dataset[i + 4][1]), int(self.dataset[i + 5][1])],
            [int(self.dataset[i + 6][1]), int(self.dataset[i + 7][1]), int(self.dataset[i + 8][1])]))
        term = Term('check3x3grid', image1, image2, image3, image4, image5, image6, image7, image8, image9, label)
        return Query(term)

    def query4x4(self, i: int):
        image1 = Term("tensor", Term(self.subset, Constant(i * 16)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 16 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 16 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 16 + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i * 16 + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i * 16 + 5)))
        image7 = Term("tensor", Term(self.subset, Constant(i * 16 + 6)))
        image8 = Term("tensor", Term(self.subset, Constant(i * 16 + 7)))
        image9 = Term("tensor", Term(self.subset, Constant(i * 16 + 8)))
        image10 = Term("tensor", Term(self.subset, Constant(i * 16 + 9)))
        image11 = Term("tensor", Term(self.subset, Constant(i * 16 + 10)))
        image12 = Term("tensor", Term(self.subset, Constant(i * 16 + 11)))
        image13 = Term("tensor", Term(self.subset, Constant(i * 16 + 12)))
        image14 = Term("tensor", Term(self.subset, Constant(i * 16 + 13)))
        image15 = Term("tensor", Term(self.subset, Constant(i * 16 + 14)))
        image16 = Term("tensor", Term(self.subset, Constant(i * 16 + 15)))
        #image1 = Term("tensor", Term(self.subset, Constant(i)))
        #image2 = Term("tensor", Term(self.subset, Constant(i + 1)))
        #image3 = Term("tensor", Term(self.subset, Constant(i + 2)))
        #image4 = Term("tensor", Term(self.subset, Constant(i + 3)))
        #image5 = Term("tensor", Term(self.subset, Constant(i + 4)))
        #image6 = Term("tensor", Term(self.subset, Constant(i + 5)))
        #image7 = Term("tensor", Term(self.subset, Constant(i + 6)))
        #image8 = Term("tensor", Term(self.subset, Constant(i + 7)))
        #image9 = Term("tensor", Term(self.subset, Constant(i + 8)))
        label = Constant(checkgridsquare([
            [int(self.dataset[i * 9][1]), int(self.dataset[i * 9 + 1][1]), int(self.dataset[i * 9 + 2][1]), int(self.dataset[i * 9 + 3][1])],
            [int(self.dataset[i * 9 + 4][1]), int(self.dataset[i * 9 + 5][1]), int(self.dataset[i * 9 + 6][1]), int(self.dataset[i * 9 + 7][1])],
            [int(self.dataset[i * 9 + 8][1]), int(self.dataset[i * 9 + 9][1]), int(self.dataset[i * 9 + 10][1]), int(self.dataset[i * 9 + 11][1])],
            [int(self.dataset[i * 9 + 12][1]), int(self.dataset[i * 9 + 13][1]), int(self.dataset[i * 9 + 14][1]), int(self.dataset[i * 9 + 15][1])],
            ]))
        #label = Constant(checkgrid3(
        #    [int(self.dataset[i][1]), int(self.dataset[i + 1][1]), int(self.dataset[i + 2][1])],
        #    [int(self.dataset[i + 3][1]), int(self.dataset[i + 4][1]), int(self.dataset[i + 5][1])],
        #    [int(self.dataset[i + 6][1]), int(self.dataset[i + 7][1]), int(self.dataset[i + 8][1])]))
        term = Term('check4x4grid', image1, image2, image3, image4, image5, image6, image7, image8, image9, image10,
                    image11, image12, image13, image14, image15, image16, label)
        return Query(term)

    def to_query(self, i: int) -> Query:
        if grid_to_query == "1x3":
            return self.query1x3(i)
        # return self.query1x3(i)
        # return self.query2x3(i)
        # return self.query2x2(i)
        # return self.query3x3(i)
        #return self.query4x4(i)

    def test_query_1x3(self, i):
        image1 = Term("tensor", Term(self.subset, Constant(i*3)))
        image2 = Term("tensor", Term(self.subset, Constant(i*3 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i*3 + 2)))

        winner = Var('winner')
        term = Term('check1x3grid', image1, image2, image3, winner)
        query = Query(term)
        actual_winner = checkgrid1(
            [int(self.dataset[i * 3][1]), int(self.dataset[i * 3 + 1][1]), int(self.dataset[i * 3 + 2][1])])
        return query, actual_winner

    def get_test_query(self, i):
        return self.test_query_1x3(i)


# grid is van vorm: [A1,A2,A3]
def checkgrid1(grid):
    first = grid[0]
    for i in range(len(grid)):
        if int(first) != int(grid[i]) or int(first) not in numbers:
            return neutral_number
    return first

def checkgrid2(row1, row2):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    for number in numbers:
        if number in results:
            return number
    return neutral_number

def checkgrid23(row1, row2):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    results += [checkgrid1([row1[0], row2[0]])]
    results += [checkgrid1([row1[1], row2[1]])]
    for number in numbers:
        if number in results:
            return number
    return neutral_number

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
    for number in numbers:
        if number in results:
            return number
    return neutral_number


def checkgrid22(row1, row2):
    results = []
    results += [checkgrid1(row1)]
    results += [checkgrid1(row2)]
    results += [checkgrid1([row1[0], row2[0]])]
    results += [checkgrid1([row1[1], row2[1]])]
    results += [checkgrid1([row1[0], row2[1]])]
    results += [checkgrid1([row1[1], row2[0]])]
    for number in numbers:
        if number in results:
            return number
    return neutral_number


def checkgridsquare(grid):
    results = []
    diagonal1 = []
    diagonal2 = []
    for i in range(len(grid)):
        results += [checkgrid1(grid[i])]
        vertical = []
        for j in range(len(grid)):
            vertical += [grid[j][i]]
            if i == j:
                diagonal1 += [grid[i][j]]
                diagonal2 += [grid[i][::-1][j]]
        results += [checkgrid1(vertical)]
    results += [checkgrid1(diagonal1)]
    results += [checkgrid1(diagonal2)]