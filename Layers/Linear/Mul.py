import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Mul(Component):
    def __init__(self):
        super().__init__()
        self.X = 0
        self.y = 0

    def forward(self, X: NDArray, Y: NDArray) -> NDArray:
        self.X = X
        self.Y = Y
        return X * Y

    def backward(self, grads: NDArray) -> NDArray:
        return (grads * self.Y, grads * self.X)
        