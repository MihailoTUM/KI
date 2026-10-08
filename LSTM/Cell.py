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

        self.X = None
        self._h = None
        self._c = None

    def forward(self, X: NDArray, _h: NDArray, _c: NDArray) -> NDArray:
        self.X = X
        self._h = _h
        self._c = _c

        self.f_gate = self.sigmoid(X @ self.w_f + _h @ self.u_f + self.b_f)
        self.i_gate = self.sigmoid(X @ self.w_i + _h @ self.u_i + self.b_i)
        self.o_gate = self.sigmoid(X @ self.w_o + _h @ self.u_o + self.b_o)
        self.c_gate = self.sigmoid(X @ self.w_c + _h @ self.u_c + self.b_c)

        self.c_state = self.f_gate * _c + self.i_gate * self.c_gate
        self.h_state = self.o_gate * self.tanh(self.c_state)

        return (self.c_state, self.h_state)

    def sigmoid(self, X: NDArray) -> NDArray:
        return 1/(1 + np.exp(-X))

    def sigmoidDeriv(self, X: NDArray) -> NDArray:
        y = self.sigmoid(X)
        return y * (1 - y)

    def tanh(self, X: NDArray) -> NDArray:
        return np.tanh(X)

    def tanhDeriv(self, X: NDArray) -> NDArray:
        return 1 - self.tanh(X)**2

    def backward(self, h_grads: NDArray, c_grads: NDArray) -> NDArray:
        dL_do = h_grads * self.tanh(self.c_state)
        self.w_o_grads += self.X.T @ (dL_do * self.sigmoidDeriv(self.o_gate)) 
        self.u_o_grads += self._h.T @ (dL_do * self.sigmoidDeriv(self.o_gate))
        self.b_o_grads += np.sum(dL_do * self.sigmoidDeriv(self.o_gate), axis=0)

        dL_dcell = h_grads * self.o_gate * self.tanhDeriv(self.c_state)

        dL_df = dL_dcell * self._c
        dL_di = dL_dcell * self.c_gate
        dL_dc = dL_dcell * self.i_gate

        self.w_f_grads += self.X.T @ (dL_df * self.sigmoidDeriv(self.f_gate))
        self.u_f_grads += self._h.T @ (dL_df * self.sigmoidDeriv(self.f_gate))
        self.b_f_grads += np.sum(dL_df * self.sigmoidDeriv(self.f_gate), axis=0)

        self.w_i_grads += self.X.T @ (dL_di * self.sigmoidDeriv(self.i_gate))
        self.u_i_grads += self._h.T @ (dL_di * self.sigmoidDeriv(self.i_gate))
        self.b_i_grads += np.sum(dL_di * self.sigmoidDeriv(self.i_gate), axis=0)
        
        self.w_c_grads += self.X.T @ (dL_dc * self.sigmoidDeriv(self.c_gate))
        self.u_c_grads += self._h.T @ (dL_dc * self.sigmoidDeriv(self.c_gate))
        self.b_c_grads += np.sum(dL_dc * self.sigmoidDeriv(self.c_gate), axis=0)

        dL_d_h = 0
        
        dL_d_h += (dL_df * self.sigmoidDeriv(self.f_gate)) @ self.u_f
        dL_d_h += (dL_di * self.sigmoidDeriv(self.i_gate)) @ self.u_i
        dL_d_h += (dL_do * self.sigmoidDeriv(self.o_gate)) @ self.u_o
        dL_d_h += (dL_dc * self.sigmoidDeriv(self.c_gate)) @ self.u_c

        return (dL_dcell * self.f_gate, dL_d_h)

