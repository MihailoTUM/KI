import numpy as np
from numpy.typing import NDArray
import torch

class Linear():
    def __init__(self, input_n, output_n, init="Normal"):
        self.weights = np.random.rand(input_n, output_n) - 0.5
        self.bias = np.random.rand(output_n)

        self.weights_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.bias)

    def forward(self, X: NDArray) -> NDArray:
        return X @ self.weights + self.bias

    def backward(self, grads):
        '''Pass-Down grads'''

        
        return

    def update(self, lr=0.1):
        self.weights -= lr * self.weights_grads
        self.bias -= lr * self.bias_grads