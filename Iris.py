import numpy as np
import kagglehub
import pandas as pd
from Models.FFN import FFN
from Loss.CrossEntropyLoss import CrossEntropyLoss
from DataLoader.DataLoader import DataLoader

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

max = np.max(X, axis=0, keepdims=True)
min = np.min(X, axis=0, keepdims=True)

X = (X - min)/(max - min)


model = FFN(4, 6, 3)
crossEntropy = CrossEntropyLoss()
data = DataLoader(X[:120], y[:120])

epochs = 50
l = 0

for epoch in range(epochs):
    for X, y in data:
        logits = model.forward(X)
        loss = crossEntropy.loss(logits, y)

        l += loss

        grads = crossEntropy.backward(logits, y)
        model.backward(grads)
        model.update(lr=0.05)
    print(f"Epoch: {epoch + 1}, Loss: {l/len(data)}")
    l = 0

print(X)

out = model.forward(X)
softmax = crossEntropy.softmax(out)

pred = np.argmax(softmax, axis=1)
true = np.argmax(y, axis=1)

correct = pred == true
accuracy = np.mean(correct)
print(f"Accuracy: {accuracy}%")