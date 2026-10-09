import numpy as np
from numpy.typing import NDArray
import torch
from Loss.CrossEntropyLoss import CrossEntropyLoss

class FFN():
    def __init__(self, dims):
        self.name = "model"
        self.input = None
        self.z = []
        self.a = []
        self.p = 0.1

        self.weights = []
        self.bias = []

        self.weights_grads = []
        self.bias_grads = []

        self.dims = dims
        self.mask = []
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
        return np.tanh(X)

    def tanhDeriv(self, X: NDArray):
        return 1 - self.tanh(X)

    # Dropout (Regulisierung)
    def dropout(self, X: NDArray, p=0.5, training=True):
        if training:
            mask = (np.random.rand(*X.shape) > self.p).astype(float)
            self.mask.append(mask)
            return (X * mask)/(1 - self.p)
        return X

    # Ableitungs der Regulisierung
    def dropoutDeriv(self, X: NDArray, mask, p=0.5):
        return mask/(1 - self.p)

    def forward(self, X: NDArray, training=True):
        self.z = []
        self.a = []
        self.mask = []

        self.input = out = X
        self.z.append(X)

        for idx in range(len(self.weights) - 1):
            out = out @ self.weights[idx] + self.bias[idx]
            self.z.append(out)
            out = self.relu(out)
            self.a.append(out)
            out = self.dropout(out, p=0.5, training=training)

        # letztes Layer ohne Aktivierung
        out = out @ self.weights[len(self.weights) - 1] + self.bias[len(self.bias) - 1]
        return out

    def backward(self, grads):
        # print(grads.shape)
        # print(self.a[len(self.a) - 1].T.shape)
        # print(self.weights_grads[len(self.weights_grads) - 1].shape)

        self.weights_grads[len(self.weights_grads) - 1] = self.a[len(self.a) - 1].T @ grads
        self.bias_grads[len(self.bias_grads) - 1] = np.sum(grads, axis=0, keepdims=False)

        # print(f"weights_grads: {self.weights_grads[len(self.weights_grads) - 1].shape}, bias_grads: { self.bias_grads[len(self.bias_grads) - 1].shape}")

        for idx in range(len(self.weights) - 1, 0, -1):
            dL_da = grads @ self.weights[idx].T * self.dropoutDeriv(self.a[idx - 1], self.mask[idx - 1])
            dL_dz = dL_da * self.reluDeriv(self.z[idx])

            # print(f"dL_da: {dL_da.shape}, dL_dz: {dL_dz.shape}")
            # print(f"a: {self.a[idx - 2].shape}, z: {self.z[idx - 1].shape}")

            if idx - 2 < 0:
                self.weights_grads[idx - 1] = self.input.T @ dL_dz
            else:
                self.weights_grads[idx - 1] = self.a[idx - 2].T @ dL_dz

            self.bias_grads[idx - 1] = np.sum(dL_dz, axis=0, keepdims=False)

            # print(f"weights_grads: {self.weights_grads[idx - 1].shape}, bias_grads: {self.bias_grads[idx - 1].shape}")

            grads = dL_dz

    def update(self, lr=0.1):
        for idx in range(len(self.weights)):
            # print(f"{self.weights[idx].shape}, {self.weights_grads[idx].shape}")
            # print(f"{self.bias[idx].shape}, {self.bias_grads[idx].shape}")
            self.weights[idx] -= lr * self.weights_grads[idx]
            self.bias[idx] -= lr * self.bias_grads[idx]

    def reset(self):
        for idx in range(len(self.weights)):
            self.weights_grads[idx] = np.zeros_like(self.weights[idx])
            self.bias_grads[idx] = np.zeros_like(self.bias[idx])
