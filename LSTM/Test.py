from LSTM.LSTM import LSTM
from LSTM.Cell import Cell
import numpy as np

lstm = LSTM(n_input=2, n_output=3, len=10)

X = np.random.rand(20, 10, 2)
out = lstm.forward(X)
print(out)
print(out.shape)