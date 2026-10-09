import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Add(Component):
    def __init__(self):
        super().__init__()

    def forward(self, X: NDArray, Y: NDArray) -> NDArray:
        return X + Y

    def backward(self, grads: NDArray) -> NDArray:
        return grads
