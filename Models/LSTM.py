import numpy as np
import torch
from typing import List

'''Ohne Autograd'''

class LSTM():
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_cell: int,
            n_output: int,
            len: int     
        ):

        self.n_input = n_input
        self.n_hidden= n_hidden
        self.n_cell = n_cell
        self.n_output = n_output

        self.w_f = np.random.rand()
        self.w_i = np.random.rand()
        self.w_c = np.random.rand()
        self.w_o = np.random.rand()

        self.w_f_grads = np.zeros_like(self.w_f)
        self.w_i_grads = np.zeros_like(self.w_i)
        self.w_c_grads = np.zeros_like(self.w_c)
        self.w_o_grads = np.zeros_like(self.w_o)

        self.b_f = np.random.rand()
        self.b_i = np.random.rand()
        self.b_c = np.random.rand()
        self.b_o = np.random.rand()

        self.b_f_grads = np.zeros_like(self.b_f)
        self.b_i_grads = np.zeros_like(self.b_i)
        self.b_c_grads = np.zeros_like(self.b_c)
        self.b_o_grads = np.zeros_like(self.b_o)

        self.h = 0
        self.len = len
        self.layers = []
        self.create()

    def create(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

    def update(self):
        pass

    def reset(self):
        pass