import numpy as np
from numpy.typing import NDArray
from Layers.Cell.Cell import Cell
from typing import List

class RNN():
    def __init__(self, n_input, n_hidden, n_output, len):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.weights_x = np.random.rand(n_input, n_hidden) * np.sqrt(1/n_input)
        self.weights_h = np.random.rand(n_hidden, n_hidden) * np.sqrt(1/n_hidden)
        self.weights_y = np.random.rand(n_hidden, n_output) * np.sqrt(1/n_hidden)

        self.bias_h = np.random.rand(n_hidden)
        self.bias_y = np.random.rand(n_output)

        self.weights_x_grads = np.zeros_like(self.weights_x)
        self.weights_h_grads = np.zeros_like(self.weights_h)
        self.weights_y_grads = np.zeros_like(self.weights_y)

        self.bias_h_grads = np.zeros_like(self.bias_h)
        self.bias_y_grads = np.zeros_like(self.bias_y)

        self.h = 0
        self.len = len
        self.layers: List[Cell]  = []
        self.create()

    def create(self):
        for i in range(self.len):
            self.layers.append(Cell(
                self.weights_x, self.weights_h, self.weights_y, self.bias_h, self.bias_y,
                self.weights_x_grads, self.weights_h_grads, self.weights_y_grads, self.bias_h_grads, self.bias_y_grads
            ))

    def forward(self, X: NDArray, training=False) -> NDArray:
        self.h = np.zeros((X.shape[1], self.n_hidden))
        h = self.h
        y = 0

        for idx in range(self.len):
            h, y = self.layers[idx].forward(X[idx], h)

        return y

    def backward(self, grads: NDArray) -> NDArray:
        _h_grads = np.zeros_like(self.h)

        for idx in range(self.len - 1, -1, -1):
            _h_grads = self.layers[idx].backward(grads[idx], _h_grads)

        return

    def update(self, lr=0.1):
        self.weights_x -= lr * self.weights_x_grads
        self.weights_h -= lr * self.weights_h_grads
        self.weights_y -= lr * self.weights_y_grads

        self.bias_y -= lr * self.bias_y_grads
        self.bias_h -= lr * self.bias_h_grads

    def reset(self):
        self.weights_x_grads.fill(0)
        self.weights_h_grads.fill(0)
        self.weights_y_grads.fill(0)
        self.bias_h_grads.fill(0)
        self.bias_y_grads.fill(0)

# X = np.random.rand(10, 1, 8)


# # model = RNN(8, 6, 4, len=10)
# # out = model.forward(X)

# # grads = np.random.rand(10, 1, 4)
# # model.backward(grads)

# # model.update()
# # model.reset()