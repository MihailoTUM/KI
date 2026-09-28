import numpy as np
from numpy.typing import NDArray

def min_max(X: NDArray) -> NDArray:
    max = np.max(X, axis=0, keepdims=True)
    min = np.min(X, axis=0, keepdims=True)

    return (X - min)/(max - min)

def standard(X: NDArray) -> NDArray:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma == 0, 1, sigma)

    return (X - mu) / sigma

