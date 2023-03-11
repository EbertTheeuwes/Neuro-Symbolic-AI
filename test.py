import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from data import MNISTImages, AdditionDataset

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

print(len(mnist_trainset))

print(torch.__version__)
#mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
#mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
#train_image_zero, train_target_zero = mnist_trainset[0]
#train_image_one, train_target_one = mnist_trainset[1]
#train_image_zero.show()
#train_image_one.show()
print(mnist_testset[321][1])
print(mnist_testset[322][1])
print(mnist_testset[323][1])
trainset = []
#for i in range(len(mnist_trainset)):
#    if int(mnist_trainset[i][1]) == 0 or int(mnist_trainset[i][1]) == 1:
#        trainset += [mnist_trainset[i]]
        #print(mnist_trainset[i][1])

for i in range(len(trainset)):
    if int(trainset[i][1]) == 0 or int(trainset[i][1]) == 1:
        print(trainset[i][1])
