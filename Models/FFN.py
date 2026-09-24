import numpy as np
from numpy.typing import NDArray
import torch
from Loss.CrossEntropyLoss import CrossEntropyLoss

class FFN():
    def __init__(self, dims):
        self.input = None
        self.z = []
        self.a = []

        self.weights = []
        self.bias = []

        self.weights_grads = []
        self.bias_grads = []

        self.dims = dims
        self.create()

    def create(self):
        for idx in range(len(self.dims) - 1):
            self.weights.append(np.random.rand(self.dims[idx], self.dims[idx + 1]) - 0.5)
            self.bias.append(np.random.rand(self.dims[idx + 1]))

            self.weights_grads.append(np.zeros(shape=(self.dims[idx], self.dims[idx + 1])))
            self.bias_grads.append(np.zeros(shape=(self.dims[idx + 1])))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    # Aktivierungsfunktion
    def relu(self, X: NDArray):
        return np.maximum(0, X)

    # Ableitung der Aktivierungsfunktion
    def reluDeriv(self, X: NDArray):
        return (X > 0).astype(float)

    def tanh(self, X: NDArray):
        return 

    def tanhDeriv(self, X: NDArray):
        return

    ##
    '''WEITERMACHEN mit Dropout'''
    ###

    # Dropout (Regulisierung)
    def dropout(self, X: NDArray, p=0.5):
        pass

    # Ableitungs der Regulisierung
    def dropoutDeriv(self, X: NDArray, p=0.5):
        pass

    def forward(self, X: NDArray):
        self.input = out = X
        self.z.append(X)

        for idx in range(len(self.weights) - 1):
            out = out @ self.weights[idx] + self.bias[idx]
            self.z.append(out)
            out = self.relu(out)
            self.a.append(out)

        # letztes Layer ohne Aktivierung
        out = out @ self.weights[len(self.weights) - 1] + self.bias[len(self.bias) - 1]
        return out

    def backward(self, grads):
        self.weights_grads[len(self.weights_grads) - 1] = self.a[len(self.a) - 1].T @ grads
        self.bias_grads[len(self.bias_grads) - 1] = np.mean(grads, axis=0, keepdims=False)

        # print(f"weights_grads: {self.weights_grads[len(self.weights_grads) - 1].shape}, bias_grads: { self.bias_grads[len(self.bias_grads) - 1].shape}")

        for idx in range(len(self.weights) - 1, 0, -1):
            dL_da = grads @ self.weights[idx].T
            dL_dz = dL_da * self.reluDeriv(self.z[idx])

            # print(f"dL_da: {dL_da.shape}, dL_dz: {dL_dz.shape}")
            # print(f"a: {self.a[idx - 2].shape}, z: {self.z[idx - 1].shape}")

            if idx - 2 < 0:
                self.weights_grads[idx - 1] = self.input.T @ dL_dz
            else:
                self.weights_grads[idx - 1] = self.a[idx - 2].T @ dL_dz

            self.bias_grads[idx - 1] = np.mean(dL_dz, axis=0, keepdims=False)

            # print(f"weights_grads: {self.weights_grads[idx - 1].shape}, bias_grads: {self.bias_grads[idx - 1].shape}")

            grads = dL_dz

    def update(self, lr=0.1):
        for idx in range(len(self.weights)):
            # print(f"{self.weights[idx].shape}, {self.weights_grads[idx].shape}")
            # print(f"{self.bias[idx].shape}, {self.bias_grads[idx].shape}")
            self.weights[idx] -= lr * self.weights_grads[idx]
            self.bias[idx] -= lr * self.bias_grads[idx]

dims = [
    784, 128, 64, 10
]

"""
    dims = [
        ["relu", 784, p=0.2],
        ["tanh", 128, p=0.5]
    ]
"""

model = FFN(dims=dims)
loss = CrossEntropyLoss()

X = np.random.rand(2, 784)

# for weight in model.get_weights():
#     print(weight.shape)

# for bias in model.get_bias():
#     print(bias.shape)

grads = np.random.rand(2, 10)

model.forward(X)
model.backward(grads)

out = model.forward(X)
# print(out)
# print(out.shape)

l = loss.softmax(out)
# print(l)
# print(l.shape)
