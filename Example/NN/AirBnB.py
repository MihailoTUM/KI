import kagglehub
import pandas as pd
import numpy as np
from NN.NN import NN
from Layers.Linear.Linear import Linear
from Layers.Activation.Relu import Relu
from Layers.Activation.Dropout import Dropout
from DataLoader.DataLoader import DataLoader
from Optimizer.SGD import SGD
from Loss.MeanSquaredLoss import MeanSquaredLoss
from numpy.typing import NDArray

path = kagglehub.dataset_download("dipeshkhemani/airbnb-cleaned-europe-dataset")
dataframe = pd.read_csv(f"{path}/Aemf1.csv")

def standard(X: NDArray) -> NDArray:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma == 0, 1, sigma)

    return (X - mu) / sigma

price = dataframe[["Price"]].to_numpy()
price_average = np.mean(price)
price_log = np.log(price)

price_mu = np.mean(price, axis=0)
price_sigma = np.std(price, axis=0)
price_sigma = np.where(price_sigma == 0, 1, price_sigma)

price_log = standard(price_log)


cities = ["Amsterdam", "Athens", "Barcelona", "Berlin", "Budapest", "Lisbon", "Paris", "Rome", "Vienna"]
days = ["Weekday", "Weekend"]
inputs = dataframe[["City", "Day", "Cleanliness Rating", "Guest Satisfaction", "Bedrooms", "City Center (km)", "Restraunt Index"]].to_numpy()

X = []
X_2 = []

for element in inputs:
    a = []
    b = []

    if element[0] == "Amsterdam":
        a.extend([1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif element[0] == "Athens":
        a.extend([0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif element[0] == "Barcelona":
        a.extend([0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif element[0] == "Berlin":
        a.extend([0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif element[0] == "Budapest":
        a.extend([0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif element[0] == "Lisbon":
        a.extend([0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif element[0] == "Paris":
        a.extend([0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif element[0] == "Rome":
        a.extend([0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif element[0] == "Vienna":
        a.extend([0, 0, 0, 0, 0, 0, 0, 0, 1])

    if element[1] == "Weekday":
        a.append(0)
    else:
        a.append(1)

    X.append(a)

    b.append(element[2])
    b.append(element[3])
    b.append(element[4])
    b.append(element[5])
    b.append(element[6])
    X_2.append(b)

X = np.array(X)
X_2 = np.array(X_2)
X_2 = standard(X_2)

result = np.hstack([X, X_2])

X_train = result[:40000]
y_train = price_log[:40000]

data = DataLoader(X_train, y_train, batch=32)

model = NN([
    Linear(15, 32),
    Relu(),
    Dropout(),
    Linear(32, 16),
    Relu(),
    Dropout(),
    Linear(16, 1) 
])



loss = MeanSquaredLoss()
optim = SGD()

optim.train(model, loss, data, lr=0.001, epochs=20, momentum=0.9, lr_decay=0.1, n=10)

X_test = result[40000:40100]
y_test = price[40000:40100]

output = model.forward(X_test, training=False)

preis_pred = np.exp(output * price_sigma + price_mu)
print(preis_pred)
print(preis_pred.shape)
preis_correct = np.exp(y_test * price_sigma + price_mu)
print(preis_correct)
print(preis_correct.shape)

acc = np.mean(np.abs(preis_pred - preis_correct))
print(acc)
print(np.mean(preis_pred))
print(price_average)
print(acc.shape)


