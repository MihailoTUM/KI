from LSTM.LSTM import LSTM
from LSTM.Cell import Cell
from Layers.Linear.Linear import Linear
import numpy as np
import matplotlib.pyplot as plt
from Loss.MeanSquaredLoss import MeanSquaredLoss

X = np.sin(np.arange(0, 1000, 0.1))
T = 10

X_train = np.array([X[i:i+T] for i in range(len(X) - T)])
y_train = np.array([X[i+T] for i in range(len(X) - T)])

X_train = X_train[:, :, None]
y_train = y_train[:, None]

batch_size = 32
num_batch = X_train.shape[0] // batch_size
index = 0

x = []
y = []

for i in range(num_batch):
    x.append(X_train[index:index + batch_size])
    y.append(y_train[index:index + batch_size])

X = np.array(x)
Y = np.array(y)

model = LSTM(1, 1, 10)
layer = Linear(1, 1)
loss = MeanSquaredLoss()


EPOCHS = 125
learning_rate = 0.01

for epoch in range(EPOCHS):
    loss_per_epoch = 0

    for batch in range(num_batch):
        out = model.forward(X[batch])
        out = layer.forward(out)
        l = loss.loss(out, Y[batch], eps=1e-05)
        loss_grads = loss.backward(out, Y[batch])
                                   
        loss_per_epoch += l
        g = layer.backward(loss_grads)
        model.backward(g)
        model.update(lr=learning_rate)
        layer.update(lr=learning_rate)
        model.reset()
        layer.reset()

    print(f"Epoch {epoch + 1}, Loss: {loss_per_epoch/X.shape[0]}")
        

X_test = X.reshape(-1, 10, 1)
y_test = Y.reshape(-1)

print(X_test.shape)
print(y_test.shape)

output = model.forward(X_test)

plt.plot(y_test, label="echt")
plt.plot(output.flatten(), label="Vorhersage", linestyle="--")
plt.legend()
plt.savefig(r"C:\KI\Example\LSTM\sinus.png", dpi=150)
plt.show()

