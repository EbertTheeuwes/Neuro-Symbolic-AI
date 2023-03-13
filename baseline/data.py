import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

full_mnist_trainset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
full_mnist_testset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)

mnist_trainset = []
for i in range(len(full_mnist_trainset)):
    if int(full_mnist_trainset[i][1]) == 5 or int(full_mnist_trainset[i][1]) == 9:
        mnist_trainset += [full_mnist_trainset[i]]
mnist_testset = []
for i in range(len(full_mnist_testset)):
    if int(full_mnist_testset[i][1]) == 5 or int(full_mnist_testset[i][1]) == 9:
        mnist_testset += [full_mnist_testset[i]]

datasets = {
    "train": mnist_trainset,
    "test": mnist_testset,
}


class TictactoeDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.originaldataset = datasets[subset]
        self.labels = []

        for i in range(0, self.__len__()):
            label = self.getLabel(i)
            self.labels.append(label)

        #for i in range(0, 2):
        #    print(self.originaldataset[i])
        #    print(self.originaldataset[i][0].shape)

    def __getitem__(self, index):
        listTensors = (self.originaldataset[index][0],self.originaldataset[index+1][0],self.originaldataset[index+2][0])
        # torch.cat(listTensors,)

        #print("shape1 ", self.originaldataset[index][0].shape)
        #print("shape2 ", self.originaldataset[index+1][0].shape)
        #print("shape3 ", self.originaldataset[index + 2][0].shape)
        #print("cat shape" , torch.cat(listTensors,1).shape)
        resulting_tensor = torch.cat(listTensors, 1)
        return resulting_tensor, self.labels[index]

    def __len__(self):
        return len(self.originaldataset) - 2

    def getLabel(self, index):
        # one hot encoding with:
        # no one wins -> 1,0,0
        # 5 wins -> 0,1,0
        # 9 wins -> 0,0,1
        firstlabel = self.originaldataset[index][1]
        for i in [1,2]:
            if self.originaldataset[index + i][1] != firstlabel:
                return torch.tensor([1., 0., 0.])

        if firstlabel == 5:
            return torch.tensor([0., 1., 0.])

        return torch.tensor([0., 0., 1.])


test = TictactoeDataset("train")
dl = DataLoader(test)


total = 0
total0 = 0
total5 = 0
total9 = 0
for i, (x,y) in enumerate(dl):
    total += 1
    if torch.argmax(y) == 0:
        total0 += 1
    elif torch.argmax(y) == 1:
        total5 += 1
    elif torch.argmax(y) == 2:
        total9 += 1

print("percentage 0 = ", total0/total*100)
print("percentage 5 = ", total5/total*100)
print("percentage 9 = ", total9/total*100)


