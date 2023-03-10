import torch

from data import MNISTImages, AdditionDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from network import MNIST_Net
from deepproblog.evaluate import get_confusion_matrix

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("model.pl", [net])
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", MNISTImages("train"))
model.add_tensor_source("test", MNISTImages("test"))

dataset = AdditionDataset("train")
testset = AdditionDataset("test")

# Train the model
loader = DataLoader(dataset, 2, False)
train = train_model(model, loader, 1, log_iter=100, profile=0)
model.save_state("snapshot/trained_model.pth")
#train.logger.comment(dumps(model.get_hyperparameters()))
train.logger.comment(
    "Accuracy {}".format(get_confusion_matrix(model, testset, verbose=1).accuracy())
)
train.logger.comment(
    "Confusion Matrix {}".format(get_confusion_matrix(model, testset, verbose=1))
)
train.logger.write_to_file("log/" + 'test')

# Query the model
for i in range(len(testset)):
    query = testset.to_query(i)
    result = model.solve([query])[0]
    print(result)


#query = dataset.to_query(0)
#result = model.solve([query])[0]
#print(result)