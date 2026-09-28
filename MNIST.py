import numpy as np
import pandas as pd
import kagglehub
from Models.FFN import FFN
from Loss.CrossEntropyLoss import CrossEntropyLoss
from DataLoader.DataLoader import DataLoader
from Optimizer.SGD import SGD

'''MNIST Dataset'''

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

model = FFN([784, 128, 64, 10])
loss = CrossEntropyLoss()
optim = SGD()

optim.train(model, loss, data, lr=0.1, epochs=15)
optim.test(X_test, y_test, model, loss)

