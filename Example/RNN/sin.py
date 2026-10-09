from RNN.RNN import RNN
from Optimizer.SGD import SGD
from Loss.CrossEntropyLoss import CrossEntropyLoss
from Loss.MeanSquaredLoss import MeanSquaredLoss
from DataLoader.DataLoader import DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt

series = np.sin(np.arange(0, 100, 0.1))   # 1000 Werte
T = 10

X = np.array([series[i:i + T] for i in range(len(series) - T)])   # (N, T)
y = np.array([series[i + T] for i in range(len(series) - T)])     # (N,)

X_train = X.T[:, :, None]    # (T, N, 1)
y_train = y[:, None]         # (N, 1)

batch_size = 32
index = 0

X_t = []
y_t = []


for i in range(X_train.shape[1] // batch_size):
    X_t.append(X_train[:, index:index + batch_size])
    y_t.append(y_train[index: index + batch_size])
    index += batch_size

X_t = np.array(X_t)
y_t = np.array(y_t)

model = RNN(1, 6, 1, len=10)
loss = MeanSquaredLoss()

epochs = 125
lr = 0.01

for epoch in range(epochs):
    loss_per_epoch = 0

    for i in range(X_t.shape[0]):
        output = model.forward(X_t[i], training=True)
        l = loss.loss(output, y_t[i], eps=1e-05)
        loss_grads = loss.backward(output, y_t[i])
        grads = np.zeros((10, *loss_grads.shape))          # (10, 31, 1)
        grads[-1] = loss_grads
        loss_per_epoch += l

        
        model.backward(grads)
        model.update(lr)
        model.reset()
    print(f"Epoch {epoch + 1}, Loss: {loss_per_epoch/X_t.shape[0]}")

X_all = X.T[:, :, None]
print(X_all.shape)
print(y.shape)

output = model.forward(X_train, training=False)

plt.plot(y, label="echt")
plt.plot(output.flatten(), label="Vorhersage", linestyle="--")
plt.legend()
plt.savefig(r"C:\KI\Example\RNN\sin.png")
plt.show()