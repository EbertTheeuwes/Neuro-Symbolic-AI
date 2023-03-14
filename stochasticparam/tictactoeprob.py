import torch
from deepproblog.query import Query
from problog.logic import Term, Constant

from dataprob import MNISTImages, AdditionDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from networkprob import MNIST_Net
from deepproblog.optimizer import SGD
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("modelprob.pl", [net])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))
model.optimizer = SGD(model, 5e-2)

dataset = AdditionDataset("train")
testset = AdditionDataset("test")

# Train the model
loader = DataLoader(dataset, 2, False)
train = train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/trained_model.pth")