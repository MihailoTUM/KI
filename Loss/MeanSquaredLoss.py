import numpy as np
import torch
from Loss.Loss import Loss
from numpy.typing import NDArray

class MeanSquaredLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, logits: NDArray):
        pass

    def backward(self, logits: NDArray, y: NDArray) -> NDArray:
        return (logits - y) / logits.shape[0]

    def loss(self, logits: NDArray, y: NDArray, eps=1e-05) -> NDArray:
        return np.mean(0.5 * (logits - y)**2, axis=0, keepdims=True)