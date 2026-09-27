from Layers.Linear.Linear import Linear
from Layers.Activation.Relu import Relu
from Layers.Activation.Dropout import Dropout
from Layers.Component.Component import Component
from Models.MLP import MLP
from typing import List
from numpy.typing import NDArray

model = MLP([
    Linear(784, 128),
    Relu(),
    Dropout(p=0.1),
    Linear(128, 64),
    Relu(),
    Dropout()
])