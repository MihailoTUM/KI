from Layers.Linear.Linear import Linear
from Layers.Activation.Relu import Relu
from Layers.Activation.Dropout import Dropout
from Layers.Component.Component import Component
from Loss.CrossEntropyLoss import CrossEntropyLoss
from Optimizer.SGD import SGD
from Models.MLP import MLP
from DataLoader.DataLoader import DataLoader
from typing import List
from numpy.typing import NDArray
import numpy as np
import pandas as pd

model = MLP([
    Linear(784, 128),
    Relu(),
    Dropout(p=0.1),
    Linear(128, 64),
    Relu(),
    Dropout(p=0.1),
    Linear(64, 10)
])

dataframe = pd.read_csv("Datasets/mnist_train.csv", header=None)

y = np.eye(10)[dataframe.iloc[:, 0].to_numpy()]
X = dataframe.iloc[:, 1:].to_numpy() / 255

# print(X.shape)
# print(y.shape)

X_train = X[:59000]
y_train = y[:59000]
# print(len(X_train))

X_test = X[59000:60000]
y_test = y[59000:60000]
# print(len(X_test))

data = DataLoader(X_train, y_train, batch=32)
loss = CrossEntropyLoss()


optim = SGD()
optim.train(model, loss, data, lr=0.01, epochs=20, momentum=0.9, lr_decay=0.1, n=10)
optim.test(X_test, y_test, model, loss)
# 96,6% accuracy