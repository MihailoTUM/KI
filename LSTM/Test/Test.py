from LSTM.LSTM import LSTM
from LSTM.Cell import Cell
import numpy as np
from Loss.MeanSquaredLoss import MeanSquaredLoss

lstm = LSTM(n_input=1, n_output=1, len=5)
loss = MeanSquaredLoss()

X = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
X_test = X[None, : , :]
y_test = np.array([[0.6]])
out = lstm.forward(X_test)

l = loss.loss(out, y_test, eps=0.00001)

EPOCHS = 10

for epoch in range(EPOCHS):
    out = lstm.forward(X_test)
    print(f"Prediction: {out}. Y: {y_test}")
    l = loss.loss(out, y_test, eps=0.00001)
    print(f"Epoch: {epoch + 1}. Loss: {l}")
    lstm.backward(loss.backward(out, y_test))
    lstm.update(lr=1)
    lstm.reset()