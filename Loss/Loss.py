import numpy as np

class Loss():
    def __init__(self):
        pass

    def forward(self, logits):
        pass

    def backward(self, logits, y):
        pass

    def loss(self, logits, y, eps=1e-05):
        pass

