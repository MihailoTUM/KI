import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Relu(Component):
    def __init__(self):
        super().__init__()
        self.input = 0

    def activate(self, X: NDArray) -> NDArray:
        return np.maximum(0, X)

    def activateDeriv(self):
        return (self.input > 0).astype(float)

    def forward(self, X: NDArray) -> NDArray:
        self.input = X
        return self.activate(X)

    def backward(self, grads: NDArray, momentum):
        return grads * self.activateDeriv()