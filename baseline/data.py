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


class TictactoeDatasetSep2x3(Dataset):
    def __init__(self, subset):
        self.subset = subset
        # every 6 images in dataset represent a grid, first 3 first row, second 3 second row
        self.original_dataset = datasets[subset]
        self.labels = []

        for i in range(self.__len__()):
            label = self.calc_label(i)
            self.labels.append(label)

    def __getitem__(self, index):
        orig_index = index*6
        image1 = self.original_dataset[orig_index][0].unsqueeze(0)
        image2 = self.original_dataset[orig_index + 1][0].unsqueeze(0)
        image3 = self.original_dataset[orig_index + 2][0].unsqueeze(0)
        image4 = self.original_dataset[orig_index + 3][0].unsqueeze(0)
        image5 = self.original_dataset[orig_index + 4][0].unsqueeze(0)
        image6 = self.original_dataset[orig_index + 5][0].unsqueeze(0)

        return torch.cat((image1, image2, image3, image4, image5, image6), 0), self.labels[index]

    def __len__(self):
        return len(self.original_dataset) // 6

    def calc_label(self, index):
        winner_first_row = self.calc_winner_row(index*6)
        winner_second_row = self.calc_winner_row(index*6 + 3)

        # 5 is given priority
        if winner_first_row == 5 or winner_second_row == 5:
            return torch.tensor([0., 1., 0.])
        if winner_first_row == 9 or winner_second_row == 9:
            return torch.tensor([0., 0., 1.])

        return torch.tensor([1., 0., 0.])

    def calc_winner_row(self, index):
        first_label_row = self.original_dataset[index][1]
        for i in [1, 2]:
            if self.original_dataset[index + i][1] != first_label_row:
                return 0

        return first_label_row

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
                fig.add_subplot(2, 3, 1)
                plt.imshow(example[0][0].squeeze())

                fig.add_subplot(2, 3, 2)
                plt.imshow(example[0][1].squeeze())

                fig.add_subplot(2, 3, 3)
                plt.imshow(example[0][2].squeeze())

                fig.add_subplot(2, 3, 4)
                plt.imshow(example[0][3].squeeze())

                fig.add_subplot(2, 3, 5)
                plt.imshow(example[0][4].squeeze())

                fig.add_subplot(2, 3, 6)
                plt.imshow(example[0][5].squeeze())

                title = "example with winner " + str(winner)
                plt.title(title)
                plt.show()

class TictactoeDatasetSep3x3(Dataset):

    def __init__(self, subset):
        self.subset = subset
        # every 6 images in dataset represent a grid, first 3 first row, second 3 second row
        self.original_dataset = datasets[subset]
        self.labels = []

        for i in range(self.__len__()):
            label = self.calc_label(i)
            self.labels.append(label)

    def __len__(self):
        return len(self.original_dataset) // 9


    def __getitem__(self, index):
        orig_index = index * 9
        image1 = self.original_dataset[orig_index][0].unsqueeze(0)
        image2 = self.original_dataset[orig_index + 1][0].unsqueeze(0)
        image3 = self.original_dataset[orig_index + 2][0].unsqueeze(0)
        image4 = self.original_dataset[orig_index + 3][0].unsqueeze(0)
        image5 = self.original_dataset[orig_index + 4][0].unsqueeze(0)
        image6 = self.original_dataset[orig_index + 5][0].unsqueeze(0)
        image7 = self.original_dataset[orig_index + 6][0].unsqueeze(0)
        image8 = self.original_dataset[orig_index + 7][0].unsqueeze(0)
        image9 = self.original_dataset[orig_index + 8][0].unsqueeze(0)

        return torch.cat((image1, image2, image3, image4, image5, image6, image7, image8, image9), 0), self.labels[index]


    def calc_label(self, index):

        winners = [self.calc_winner_row(index * 9), self.calc_winner_row(index * 9 + 3),
                   self.calc_winner_row(index * 9 + 6), self.calc_winner_column(index * 9, 3),
                   self.calc_winner_column(index * 9 + 1, 3), self.calc_winner_column(index * 9 + 2, 3),
                   self.calc_winner_first_diagonal(index * 9, 3), self.calc_winner_second_diagonal(index * 9, 3)]

        # 5 is given priority
        if 5 in winners:
            return torch.tensor([0., 1., 0.])
        if 9 in winners:
            return torch.tensor([0., 0., 1.])

        return torch.tensor([1., 0., 0.])

    def calc_winner_row(self, index):
        first_label_row = self.original_dataset[index][1]
        for i in [1, 2]:
            if self.original_dataset[index + i][1] != first_label_row:
                return 0

        return first_label_row

    def calc_winner_column(self, index, length_row):
        first_label_column = self.original_dataset[index][1]
        for i in [1, 2]:
            if self.original_dataset[index + length_row*i][1] != first_label_column:
                return 0

        return first_label_column

    def calc_winner_first_diagonal(self, index, length_row):
        first_label = self.original_dataset[index][1]
        for i in [1,2]:
            if self.original_dataset[index + length_row*i + i][1] != first_label:
                return 0
        return first_label

    def calc_winner_second_diagonal(self, index, length_row):
        first_label = self.original_dataset[index + (length_row - 1)][1]
        for i in [1,2]:
            if self.original_dataset[index + (length_row - 1) + (length_row - 1)*i][1] != first_label:
                return 0
        return first_label

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
                fig.add_subplot(3, 3, 1)
                plt.imshow(example[0][0].squeeze())

                fig.add_subplot(3, 3, 2)
                plt.imshow(example[0][1].squeeze())

                fig.add_subplot(3, 3, 3)
                plt.imshow(example[0][2].squeeze())

                fig.add_subplot(3, 3, 4)
                plt.imshow(example[0][3].squeeze())

                fig.add_subplot(3, 3, 5)
                plt.imshow(example[0][4].squeeze())

                fig.add_subplot(3, 3, 6)
                plt.imshow(example[0][5].squeeze())

                fig.add_subplot(3, 3, 7)
                plt.imshow(example[0][6].squeeze())

                fig.add_subplot(3, 3, 8)
                plt.imshow(example[0][7].squeeze())

                fig.add_subplot(3, 3, 9)
                plt.imshow(example[0][8].squeeze())

                title = "example with winner " + str(winner)
                plt.title(title)
                plt.show()


# test = TictactoeDatasetSep3x3("train")
# test.print_images_of_winner(5)


# test = TictactoeDatasetSep("train")
# test.print_percentages()
# test.halve_no_winner_examples()
# test.print_images_of_winner(5)

# testimg = test.__getitem__(120)[0]
# print(testimg.shape)
# plt.imshow(testimg.squeeze())
# plt.show()

# test.print_images_of_winner(0)


# code to find index at which example where 5 wins
# test = TictactoeDatasetSep("train")
# dl = DataLoader(test, batch_size=None)
#
# for i, (img, label) in enumerate(dl):
#     if torch.argmax(label) == 1:
#         print(img.shape)
#         print("first example where 5 wins index: ", i)
#         fig = plt.figure(figsize=(8, 8))
#         fig.add_subplot(1, 3, 1)
#         plt.imshow(img[0].squeeze())
#         fig.add_subplot(1, 3, 2)
#         plt.imshow(img[1].squeeze())
#         fig.add_subplot(1, 3, 3)
#         plt.imshow(img[2].squeeze())
#         title = "example with winner " + str(5)
#         plt.title(title)
#         plt.show()
#         break
#

# def calc_label(self, index):
#     winner_first_row = self.calc_winner_row(index * 3)
#     winner_second_row = self.calc_winner_row(index * 3 + 3)
#     winner = 0
#
#     if winner_first_row == 0:
#         winner = winner_second_row
#
#     elif winner_second_row == 0:
#         winner = winner_first_row
#
#     else:
#         return None
#
#     # one hot encoding with:
#     # no one wins -> 1,0,0
#     # 5 wins -> 0,1,0
#     # 9 wins -> 0,0,1
#     if winner == 0:
#         return torch.tensor([1., 0., 0.])
#     if winner == 5:
#         return torch.tensor([0., 1., 0.])
#
#     return torch.tensor([0., 0., 1.])

