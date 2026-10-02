import numpy as np
from numpy.typing import NDArray

class LSTMCell():
    def __init__(self):
        pass

    def forward(self, X: NDArray, _h: NDArray) -> NDArray:
        pass

    def backward(self):
        pass

    def sigmoid(self, X: NDArray):
        return 1/(1 + np.exp(-X))