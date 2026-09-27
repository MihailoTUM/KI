from Layers.Activation.Relu import Relu
from Layers.Linear.Linear import Linear
import numpy as np

X = np.random.rand(10, 8)

layer_1 = Linear(8, 4)
relu_1 = Relu()
layer_2 = Linear(4, 2)
relu_2 = Relu()
layer_3 = Linear(2, 1)

out = layer_1.forward(X)
out = relu_1.forward(out)
out = layer_2.forward(out)
out = relu_2.forward(out)
out = layer_3.forward(out)

print(out)
print(out.shape)

out = layer_3.backward(np.random.rand(10, 1))
out = relu_2.backward(out)
out = layer_2.backward(out)
out = relu_1.backward(out)
out = layer_1.backward(out)

print(out)
print(out.shape)
