from typing import Mapping, Iterator

import torch
import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, Constant

from deepproblog.dataset import Dataset
from deepproblog.query import Query

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root='data/', train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root='data/', train=False, download=True, transform=transform
    ),
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
        return len(self.dataset) // 6

    def to_query(self, i: int) -> Query:
        image1 = Term("tensor", Term(self.subset, Constant(i * 6)))
        image2 = Term("tensor", Term(self.subset, Constant(i * 6 + 1)))
        image3 = Term("tensor", Term(self.subset, Constant(i * 6 + 2)))
        image4 = Term("tensor", Term(self.subset, Constant(i * 6 + 3)))
        image5 = Term("tensor", Term(self.subset, Constant(i * 6 + 4)))
        image6 = Term("tensor", Term(self.subset, Constant(i * 6 + 5)))
        #label = Constant(int(self.dataset[i*6][1]))
        if int(self.dataset[i*6][1]) % 2 == 0:
            label = Constant(int(0))
        else:
            label = Constant(int(1))
        #label = Constant(int(self.dataset[i*6][1]) + int(self.dataset[i*6 + 1][1]) + int(self.dataset[i*6 + 2][1]))
        #label = Constant(int(self.dataset[i*6][1]) + int(self.dataset[i*6][1]) + int(self.dataset[i*6][1]) +
        #    int(self.dataset[i*6][1]) + int(self.dataset[i*6 + 1][1]) + int(self.dataset[i*6 + 2][1]) +
        #    int(self.dataset[i*6 + 3][1]) + int(self.dataset[i*6 + 4][1]) + int(self.dataset[i*6 + 5][1]))
        #label = Constant(int(self.dataset[i*6][1]))
        #label = Constant(int(self.dataset[i*6][1] + int(self.dataset[i*6 + 1][1])))
        #term = Term('addition', image1, image2, label)
        #term = Term('check', [[image1,image1,image1],[image1,image2,image3],[image4,image5,image6]], label)
        term = Term('check', image1, image1, image1, image1, image2, image3, image4, image5, image6, label)
        #term = Term('add', image1, image1, image1, image1, image2, image3, image4, image5, image6, label)
        #term = Term('evenOrOdd', image1, label)
        #term = Term('idd', image1, label)
        #term = Term('add3', image1, image2, image3, label)
        #term = Term('labelnum', image1, label)
        #print(image1)
        #print(image2)
        #print(label)
        return Query(term)