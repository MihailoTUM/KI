import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Sigmoid():
    def __init__(self):
        super().__init__()
        self.input = 0

    def activate(self, X: NDArray) -> NDArray:
        return 1/(1 + np.exp(-X))

    def activateDeriv(self):
        y = self.sigmoid(self.input)
        return y * (1 - y)

    def forward(self, X: NDArray) -> NDArray:
        self.input = X
        return self.activate(X)

    def backward(self, grads: NDArray, momentum):
        return grads * self.activateDeriv()