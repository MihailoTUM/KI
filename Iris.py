import numpy as np
import kagglehub
import pandas as pd
from Models.NN import NN
from Models.FFN import FFN
from Loss.CrossEntropyLoss import CrossEntropyLoss
from DataLoader.DataLoader import DataLoader
from DataLoader.Preprocess import min_max
from Optimizer.GD import Optimizer

path = kagglehub.dataset_download("uciml/iris")

data = pd.read_csv(f"{path}/Iris.csv")
# print(data.head())

X = np.array(data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]])
y = np.array(data["Species"])

for element in range(y.shape[0]):
    if y[element] == "Iris-setosa":
        y[element] = 0
    elif y[element] == "Iris-versicolor":
        y[element] = 1
    else:
        y[element] = 2

y = np.eye(3)[y.astype(int)]

X = min_max(X)

model = NN(4, 6, 3)
model_2 = FFN([4, 6, 3])

crossEntropy = CrossEntropyLoss()
data = DataLoader(X[:125], y[:125])

optimizer = Optimizer()

optimizer.train(model, crossEntropy, data, lr=0.01, epochs=100)
optimizer.test(X, y, model, crossEntropy)

optimizer.train(model_2, crossEntropy, data, lr=0.01, epochs=100)
optimizer.test(X, y, model_2, crossEntropy)
