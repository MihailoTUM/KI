from Layers.Activation.Softmax import Softmax
import numpy as np

X = np.random.rand(2, 10)

softmax = Softmax()
out = softmax.forward(X)
