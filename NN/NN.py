from Layers.Linear.Linear import Linear
from Layers.Activation.Relu import Relu
from Layers.Activation.Dropout import Dropout
from Layers.Component.Component import Component
from typing import List
from numpy.typing import NDArray

class NN():
    def __init__(self, layers: List[Component]):
        self.layers = layers
        self.input = None

    def forward(self, X: NDArray, training=True) -> NDArray:
        self.input = out = X

        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grads: NDArray, momentum=0) -> None:
        out = grads
        for layer in reversed(self.layers):
            out = layer.backward(out, momentum)

    def update(self, lr=0.1):
        for layer in self.layers:
            layer.update(lr)

    def reset(self):
        for layer in self.layers:
            layer.reset()