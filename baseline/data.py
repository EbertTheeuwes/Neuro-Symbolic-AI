import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


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


class TictactoeDatasetSep(Dataset):
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
        orig_index = index*3
        resulting_tensor = torch.cat((self.originaldataset[orig_index][0].unsqueeze(0), self.originaldataset[orig_index+1][0].unsqueeze(0), self.originaldataset[orig_index+2][0].unsqueeze(0)), 0)
        # torch.cat(listTensors,)

        #print("shape1 ", self.originaldataset[index][0].shape)
        #print("shape2 ", self.originaldataset[index+1][0].shape)
        #print("shape3 ", self.originaldataset[index + 2][0].shape)
        #print("cat shape" , torch.cat(listTensors,1).shape)

        # resulting_tensor = torch.cat(tensors, 2)
        return resulting_tensor, self.labels[index]

    def __len__(self):
        return len(self.originaldataset)//3

    def getLabel(self, index):
        # one hot encoding with:
        # no one wins -> 1,0,0
        # 5 wins -> 0,1,0
        # 9 wins -> 0,0,1
        orig_index = index*3
        firstlabel = self.originaldataset[orig_index][1]
        for i in [1,2]:
            if self.originaldataset[orig_index + i][1] != firstlabel:
                return torch.tensor([1., 0., 0.])

        if firstlabel == 5:
            return torch.tensor([0., 1., 0.])

        return torch.tensor([0., 0., 1.])

    def print_images_of_winner(self, winner):

        index_label = None
        if winner == 0:
            index_label = 0
        if winner == 5:
            index_label = 1
        if winner == 9:
            index_label = 2
        for i in range(self.__len__()):
            example = self.__getitem__(i)
            if torch.argmax(example[1], dim=0) == index_label:

                fig = plt.figure(figsize=(8,8))
                fig.add_subplot(1, 3, 1)
                plt.imshow(example[0][0].squeeze())

                fig.add_subplot(1, 3, 2)
                plt.imshow(example[0][1].squeeze())

                fig.add_subplot(1, 3, 3)
                plt.imshow(example[0][2].squeeze())

                title = "example with winner " + str(winner)
                plt.title(title)
                plt.show()

    def print_percentages(self):
        dl = DataLoader(self)

        total = 0
        total0 = 0
        total5 = 0
        total9 = 0
        for i, (x, y) in enumerate(dl):
            total += 1
            if torch.argmax(y) == 0:
                total0 += 1
            elif torch.argmax(y) == 1:
                total5 += 1
            elif torch.argmax(y) == 2:
                total9 += 1

        print("percentage 0 = ", total0 / total * 100)
        print("percentage 5 = ", total5 / total * 100)
        print("percentage 9 = ", total9 / total * 100)



class TictactoeDatasetCat(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.originaldataset = datasets[subset]
        self.labels = []

        for i in range(self.__len__()):
            label = self.getLabel(i)
            self.labels.append(label)

        #for i in range(0, 2):
        #    print(self.originaldataset[i])
        #    print(self.originaldataset[i][0].shape)

    def __getitem__(self, index):
        orig_index = index*3
        listTensors = (self.originaldataset[orig_index][0], self.originaldataset[orig_index+1][0],self.originaldataset[orig_index+2][0])
        # torch.cat(listTensors,)

        #print("shape1 ", self.originaldataset[index][0].shape)
        #print("shape2 ", self.originaldataset[index+1][0].shape)
        #print("shape3 ", self.originaldataset[index + 2][0].shape)
        #print("cat shape" , torch.cat(listTensors,1).shape)
        resulting_tensor = torch.cat(listTensors, 2)
        return resulting_tensor, self.labels[index]

    def __len__(self):
        return len(self.originaldataset) // 3

    def getLabel(self, index):
        # one hot encoding with:
        # no one wins -> 1,0,0
        # 5 wins -> 0,1,0
        # 9 wins -> 0,0,1
        orig_index = index*3
        firstlabel = self.originaldataset[orig_index][1]
        for i in [1,2]:
            if self.originaldataset[orig_index + i][1] != firstlabel:
                return torch.tensor([1., 0., 0.])

        if firstlabel == 5:
            return torch.tensor([0., 1., 0.])

        return torch.tensor([0., 0., 1.])

    def print_images_of_winner(self, winner):

        index_label = None
        if winner == 0:
            index_label = 0
        if winner == 5:
            index_label = 1
        if winner == 9:
            index_label = 2
        for i in range(self.__len__()):
            example = self.__getitem__(i)
            if torch.argmax(example[1], dim=0) == index_label:

                plt.imshow(example[0].squeeze())
                title = "example with winner " + str(winner)
                plt.title(title)
                plt.show()

    # does not work properly
    def halve_no_winner_examples(self):
        delete = True
        i = 0
        cur_len = self.__len__()
        while i < cur_len:
            example = self.__getitem__(i)
            if torch.argmax(example[1], dim=0) == 0:
                if delete:
                    self.labels.pop(i)
                    self.originaldataset.pop(i*3)
                    self.originaldataset.pop(i*3 + 1)
                    self.originaldataset.pop(i*3 + 2)
                    delete = False
                    cur_len -= 1
                else:
                    delete = True
                    i += 1
            else:
                i += 1

    def print_percentages(self):
        dl = DataLoader(self)

        total = 0
        total0 = 0
        total5 = 0
        total9 = 0
        for i, (x, y) in enumerate(dl):
            total += 1
            if torch.argmax(y) == 0:
                total0 += 1
            elif torch.argmax(y) == 1:
                total5 += 1
            elif torch.argmax(y) == 2:
                total9 += 1

        print("percentage 0 = ", total0 / total * 100)
        print("percentage 5 = ", total5 / total * 100)
        print("percentage 9 = ", total9 / total * 100)


# test = TictactoeDatasetSep("train")
# test.print_percentages()
# test.halve_no_winner_examples()
# test.print_images_of_winner(5)

# testimg = test.__getitem__(120)[0]
# print(testimg.shape)
# plt.imshow(testimg.squeeze())
# plt.show()

# test.print_images_of_winner(0)
