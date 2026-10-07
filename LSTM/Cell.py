import numpy as np
from numpy.typing import NDArray

class Cell():
    def __init__(
            self, 
            w_f,
            w_i,
            w_c,
            w_o,
            w_f_grads,
            w_i_grads,
            w_c_grads,
            w_o_grads,
            u_f,
            u_i,
            u_c,
            u_o,
            u_f_grads,
            u_i_grads,
            u_c_grads,
            u_o_grads,
            b_f,
            b_i,
            b_c,
            b_o,
            b_f_grads,
            b_i_grads,
            b_c_grads,
            b_o_grads
        ):
        
        # WEIGHTS FOR x_t
        self.w_f = w_f
        self.w_i = w_i
        self.w_c = w_c
        self.w_o = w_o

        self.w_f_grads = w_f_grads
        self.w_i_grads = w_i_grads
        self.w_c_grads = w_c_grads
        self.w_o_grads = w_o_grads

        # WEIGHTS FOR _h
        self.u_f = u_f
        self.u_i = u_i
        self.u_c = u_c
        self.u_o = u_o

        self.u_f_grads = u_f_grads
        self.u_i_grads = u_i_grads
        self.u_c_grads = u_c_grads
        self.u_o_grads = u_o_grads

        self.b_f = b_f
        self.b_i = b_i
        self.b_c = b_c
        self.b_o = b_o

        self.b_f_grads = b_f_grads
        self.b_i_grads = b_i_grads
        self.b_c_grads = b_c_grads
        self.b_o_grads = b_o_grads

        self.f_gate = 0
        self.i_gate = 0
        self.o_gate = 0
        self.c_gate = 0
        self.c_state = 0
        self.h_state = 0

    def forward(self, X: NDArray, _h: NDArray, _c: NDArray) -> NDArray:
        self.f_gate = self.sigmoid(X @ self.w_f + _h @ self.u_f + self.b_f)
        self.i_gate = self.sigmoid(X @ self.w_i + _h @ self.u_i + self.b_i)
        self.o_gate = self.sigmoid(X @ self.w_o + _h @ self.u_o + self.b_o)
        self.c_gate = self.sigmoid(X @ self.w_c + _h @ self.u_c + self.b_c)

        self.c_state = self.f_gate * _c + self.i_gate * self.c_gate
        self.h_state = self.o_gate * self.tanh(self.c_state)

        return (self.c_state, self.h_state)

    def sigmoid(self, X: NDArray) -> NDArray:
        return 1/(1 + np.exp(-X))

    def tanh(self, X: NDArray) -> NDArray:
        return np.tanh(X)

    def backward(self, grads: NDArray) -> NDArray:
        pass