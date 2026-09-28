import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Linear(Component):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.weights = np.random.rand(input_n, output_n) - 0.5
        self.bias = np.random.rand(output_n) - 0.5

        self.weights_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.bias)

        self.weights_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.bias)

        self.input = 0

    def forward(self, X: NDArray) -> NDArray:
        # X: (exp, input), return: (exp, output)
        self.input = X
        return X @ self.weights + self.bias

    def backward(self, grads: NDArray, momentum=0) -> NDArray:
        '''Pass-Down grads'''
        # X: (exp, input), w: (input, output), return: (exp, output)

        self.weights_grads = self.input.T @ grads
        self.bias_grads = np.sum(grads, axis=0, keepdims=False)

        self.weights_velocity = momentum * self.weights_velocity + self.weights_grads
        self.bias_velocity = momentum * self.bias_velocity + self.bias_grads
        
        return grads @ self.weights.T

    def reset(self):
        self.weights_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.bias)

    def update(self, lr=0.1):
        self.weights -= lr * self.weights_velocity
        self.bias -= lr * self.bias_velocity