import numpy as np
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("uciml/iris")
data = pd.read_csv(f"{path}/Iris.csv")
# print(data.head())

class DataLoader():
    def __init__(self, X, y, batch=32):
        self.index = 0
        self.batch = batch
        self.X = X
        self.y = y

    def __iter__(self):
        self.index = 0
        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation]
        self.y = self.y[permutation]
        return self

    def __next__(self):
        if self.index + self.batch >= self.X.shape[0]:
            raise StopIteration

        end = self.index + self.batch
        p = (self.X[self.index:end], self.y[self.index:end])
        self.index = end

        return p

    def __len__(self):
        return self.X.shape[0] // self.batch

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

max = np.max(X, axis=0, keepdims=True)
min = np.min(X, axis=0, keepdims=True)

X = (X - min)/(max - min)

data = DataLoader(X, y)


