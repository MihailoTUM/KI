import numpy as np
from LSTM.Cell import Cell
from typing import List
from numpy.typing import NDArray

class LSTM():
    def __init__(
            self,
            n_input: int,
            n_output: int,
            len: int     
        ):

        self.n_input = n_input
        self.n_output = n_output

        # WEIGHTS FOR x_t
        self.w_f = (np.random.rand(n_input, n_output) - 0.5) * np.sqrt(1/n_input)
        self.w_i = (np.random.rand(n_input, n_output) - 0.5) * np.sqrt(1/n_input)
        self.w_c = (np.random.rand(n_input, n_output) - 0.5) * np.sqrt(1/n_input)
        self.w_o = (np.random.rand(n_input, n_output) - 0.5) * np.sqrt(1/n_input)

        self.w_f_grads = np.zeros_like(self.w_f)
        self.w_i_grads = np.zeros_like(self.w_i)
        self.w_c_grads = np.zeros_like(self.w_c)
        self.w_o_grads = np.zeros_like(self.w_o)

        # WEIHGTS FOR _h
        self.u_f = (np.random.rand(n_output, n_output) - 0.5) * np.sqrt(1/n_output)
        self.u_i = (np.random.rand(n_output, n_output) - 0.5) * np.sqrt(1/n_output)
        self.u_c = (np.random.rand(n_output, n_output) - 0.5) * np.sqrt(1/n_output)
        self.u_o = (np.random.rand(n_output, n_output) - 0.5) * np.sqrt(1/n_output)

        self.u_f_grads = np.zeros_like(self.u_f)
        self.u_i_grads = np.zeros_like(self.u_i)
        self.u_c_grads = np.zeros_like(self.u_c)
        self.u_o_grads = np.zeros_like(self.u_o)

        self.b_f = np.random.rand(n_output)
        self.b_i = np.random.rand(n_output)
        self.b_c = np.random.rand(n_output)
        self.b_o = np.random.rand(n_output)

        self.b_f_grads = np.zeros_like(self.b_f)
        self.b_i_grads = np.zeros_like(self.b_i)
        self.b_c_grads = np.zeros_like(self.b_c)
        self.b_o_grads = np.zeros_like(self.b_o)

        self.len: int = len
        self.layers: list[Cell] = []
        self.create()

    def create(self):
        for i in range(self.len):
            self.layers.append(Cell(
                w_f=self.w_f,
                w_i=self.w_i,
                w_c=self.w_c,
                w_o=self.w_o,
                w_f_grads=self.w_f_grads,
                w_i_grads=self.w_i_grads,
                w_c_grads=self.w_c_grads,
                w_o_grads=self.w_o_grads,
                u_f=self.u_f,
                u_i=self.u_i,
                u_c=self.u_c,
                u_o=self.u_o,
                u_f_grads=self.u_f_grads,
                u_i_grads=self.u_i_grads,
                u_c_grads=self.u_c_grads,
                u_o_grads=self.u_o_grads,
                b_f=self.b_f,
                b_i=self.b_i,
                b_c=self.b_c,
                b_o=self.b_o,
                b_f_grads=self.b_f_grads,
                b_i_grads=self.b_i_grads,
                b_c_grads=self.b_c_grads,
                b_o_grads=self.b_o_grads
            ))

    def forward(self, X: NDArray):
        _c = np.zeros(shape=(X.shape[0], self.n_output))
        _h = np.zeros(shape=(X.shape[0], self.n_output))

        for i in range(self.len):
            _c, _h = self.layers[i].forward(X[:, i, :], _h, _c)

        return

    def backward(self, grads: NDArray):
        pass

    def update(self, lr=0.1):
        self.w_f -= lr * self.w_f_grads
        self.w_i -= lr * self.w_i_grads
        self.w_c -= lr * self.w_c_grads
        self.w_o -= lr * self.w_o_grads

        self.u_f -= lr * self.u_f_grads
        self.u_i -= lr * self.u_i_grads
        self.u_c -= lr * self.u_c_grads
        self.u_o -= lr * self.u_o_grads

        self.b_f -= lr * self.b_f_grads
        self.b_i -= lr * self.b_i_grads
        self.b_c -= lr * self.b_c_grads
        self.b_o -= lr * self.b_o_grads

    def reset(self):
        self.w_f_grads.fill(0)
        self.w_i_grads.fill(0)
        self.w_c_grads.fill(0)
        self.w_o_grads.fill(0)

        self.u_f_grads.fill(0)
        self.u_i_grads.fill(0)
        self.u_c_grads.fill(0)
        self.u_o_grads.fill(0)

        self.b_f_grads.fill(0)
        self.b_i_grads.fill(0)
        self.b_c_grads.fill(0)
        self.b_o_grads.fill(0)