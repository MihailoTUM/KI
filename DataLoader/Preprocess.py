import numpy as np
from numpy.typing import NDArray

def min_max(X:NDArray) -> NDArray:
    max = np.max(X, axis=0, keepdims=True)
    min = np.min(X, axis=0, keepdims=True)

    return (X - min)/(max - min)

