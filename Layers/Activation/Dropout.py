import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Dropout(Component):
    def __init__(self, p=0.1):
        super().__init__()
        self.input = 0
        self.p = p
        self.mask = 0

    def activate(self, X: NDArray) -> NDArray:
        self.mask = (np.random.rand(*X.shape) > self.p).astype(float)
        return (X * self.mask)/(1 - self.p)

    def activateDeriv(self):
        return self.mask/(1 - self.p)

    def forward(self, X: NDArray) -> NDArray:
        self.input = X
        return self.activate(X)

    def backward(self, grads: NDArray):
        return grads * self.activateDeriv()