import numpy as np
import kagglehub
import pandas as pd
from Models.FFN import FFN
from Loss.CrossEntropyLoss import CrossEntropyLoss

path = kagglehub.dataset_download("uciml/iris")

data = pd.read_csv(f"{path}/Iris.csv")
print(data.head())

X_un = []
y_un = []

X = np.array()
y = np.array()

model = FFN(4, 4, 3)
crossEntropy = CrossEntropyLoss()

epochs = 100

for epoch in range(epochs):
    logits = model.forward(X)
    loss = crossEntropy.loss(logits, y)

    grads = crossEntropy.backward(logits, y)
    model.backward(grads)
    model.update()


# X = np.random.rand(2, 8)
# y = np.eye(4)[np.array([3, 1])]

# logits = model.forward(X)
# print(logits)
# print("\n")
# prob = crossEntropy.softmax(logits)
# print(prob)
# print("\n")
# loss = crossEntropy.loss(prob, y)
# print(loss)
# print("\n")
# grads = crossEntropy.backward(prob, y)
# print(grads)
# print("\n")

# model.backward(grads)
# model.update()

