import numpy as np
from numpy.typing import NDArray

class Component():
    def __init__(self):
        pass

    def forward(self, X: NDArray) -> NDArray:
        pass

    def backward(self, grads: NDArray) -> NDArray:
        pass

    def update(self, lr=0.1):
        pass

    def activate(self, X: NDArray) -> NDArray:
        pass

    def activateDeriv(self):
        pass

    def reset(self):
        pass