import torch
from deepproblog.query import Query
from problog.logic import Term, Constant, Var
import matplotlib.pylab as plt

from data import MNISTImages, AdditionDataset
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from network import MNIST_Net
from deepproblog.evaluate import get_confusion_matrix, get_fact_accuracy

network = MNIST_Net()
net = Network(network, "mnist_net", batching=True)
net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

model = Model("model.pl", [net])
#model = Model("model01.pl", [net])
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
#train.logger.comment(
#    "Accuracy {}".format(get_confusion_matrix(model, testset, verbose=1).accuracy())
#)
#train.logger.comment(
#    "Confusion Matrix {}".format(get_confusion_matrix(model, testset, verbose=1))
#)
#train.logger.comment(
#    "Fact accuracy {}".format(get_fact_accuracy(model, testset, verbose=1))
#)
#train.logger.comment(
#    "Fact accuracy {}".format(get_fact_accuracy(model, testset, verbose=1).accuracy())
#)
#train.logger.write_to_file("log/" + 'test')



            # Actual 0  5  9
confusion_matrix = [[0, 0, 0], # pred 0
                    [0, 0, 0], # pred 5
                    [0, 0, 0]] # pred 9
correct = 0
total = 0

# for i in range(len(testset)):
#     query, winner = testset.get_test_query(i)
#     result = model.solve([query])[0].result
#     pred_prob0 = 0
#     pred_prob5 = 0
#     pred_prob9 = 0
#     for key in result:
#         arguments = key.args
#         probability = result[key]
#         if 5 in arguments:
#             if torch.is_tensor(probability):
#                 pred_prob5 = probability.item()
#             else:
#                 pred_prob5 = probability
#         elif 9 in arguments:
#             if torch.is_tensor(probability):
#                 pred_prob9 = probability.item()
#             else:
#                 pred_prob9 = probability
#         elif 0 in arguments:
#             if torch.is_tensor(probability):
#                 pred_prob0 = probability.item()
#             else:
#                 pred_prob0 = probability
#
#     index = 0
#     if winner == 5:
#         index = 1
#     if winner == 9:
#         index = 2
#
#     predicted_prob = [pred_prob0, pred_prob5, pred_prob9]
#     max_prob = max(predicted_prob)
#     row = predicted_prob.index(max_prob)
#     confusion_matrix[row][index] += 1
#
#     if row == index:
#         correct += 1
#     total += 1
#
# accuracy = correct/total * 100
# print("Accuracy from own calculation = ", accuracy, "%")
# print("Own confusion matrix ", confusion_matrix )
#
# train.logger.comment(
#     "Accuracy {}".format(get_confusion_matrix(model, testset, verbose=1).accuracy())
# )
# print(train.logger.log_data)
# print(train.logger.log_data["time"])
#
# lists = sorted(train.logger.log_data["time"].items())  # sorted by key, return a list of tuples
#
# x, y = zip(*lists)  # unpack a list of pairs into two tuples
#
# plt.plot(x, y)
# plt.show()





# Query the model
# for i in range(len(testset)):
#     query = testset.to_query(i)
#     result = model.solve([query])[0]
#     print(result)



#query = dataset.to_query(0)
#result = model.solve([query])[0]
#print(result)

# i = 9
# image1 = Term("tensor", Term(testset.subset, Constant(i)))
# image2 = Term("tensor", Term(testset.subset, Constant(i + 1)))
# image3 = Term("tensor", Term(testset.subset, Constant(i + 2)))
#
# winner = Var('winner')
# term = Term('check1x3grid', image1, image2, image3, winner)
# query = Query(term)
# result = model.solve([query])[0].result
# print(result)
#
# for key in result:
#     arguments = key.args
#     if 5 in arguments:
#         print("probability 5 = ", result[key].item())
#     if 9 in arguments:
#         print("probabilty 9 = ", result[key].item())
#     if 0 in arguments:
#         print("probabilty 0 = ", result[key].item())