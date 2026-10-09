import numpy as np
from numpy.typing import NDArray
from Layers.Component.Component import Component

class Softmax(Component):
    def __init__(self):
        super().__init__()

    def activate(self, logits: NDArray) -> NDArray:
        max = np.max(logits, axis=1, keepdims=True)
        logits = logits - max
        exp = np.exp(logits)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

    def forward(self, logits):
        return self.activate(logits)
