import numpy as np
from numpy.typing import NDArray

class Cell():
    def __init__(self, w_x, w_h, w_y, b_h, b_y, w_x_grads, w_h_grads, w_y_grads, b_h_grads, b_y_grads):
        self.w_x = w_x
        self.w_h = w_h
        self.w_y = w_y

        self.b_h = b_h
        self.b_y = b_y

        self.w_x_grads = w_x_grads
        self.w_h_grads = w_h_grads
        self.w_y_grads = w_y_grads

        self.b_h_grads = b_h_grads
        self.b_y_grads = b_y_grads

        self._h_grads = None

        self.input = None
        self._h = None
        self.h = np.zeros(w_x.shape[1])
        self.y = np.zeros(w_y.shape[1])

    def forward(self, X: NDArray, _h: NDArray) -> NDArray:
        self.input = X
        self._h = _h
        self.h = self.tanh(X @ self.w_x + _h @ self.w_h + self.b_h)
        return (self.h, self.output())

    def output(self):
        self.y = self.h @ self.w_y + self.b_y
        return self.y

    def tanh(self, X: NDArray) -> NDArray:
        return np.tanh(X)

    def tanhDeriv(self, X: NDArray) -> NDArray:
        return 1 - np.tanh(X)**2

    def backward(self, grads, _h_grads):
        self.w_y_grads += self.h.T @ grads
        self.b_y_grads += np.sum(grads, axis=0)

        add = (grads @ self.w_y.T) + _h_grads
        tween = add * (1 - self.h**2)

        self.w_x_grads += self.input.T @ tween
        self.w_h_grads += self._h.T @ tween
        self.b_h_grads += np.sum(tween, axis=0)

        self._h_grads = tween @ self.w_h.T
        return self._h_grads