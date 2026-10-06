import numpy as np
from numpy.typing import NDArray

class LSTMCell():
    def __init__(self):
        self.weights = 0

    def forward(self, X: NDArray, _h: NDArray, _c: NDArray) -> NDArray:
        # _h = hidden state
        # _c = cell state



        return

    def backward(self):
        pass

    def sigmoid(self, X: NDArray):
        return 1/(1 + np.exp(-X))

    def tanh(self, X: NDArray):
        return