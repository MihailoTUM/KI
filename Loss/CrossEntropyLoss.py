import numpy as np
import torch
from Loss.Loss import Loss

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        max = np.max(logits, axis=1, keepdims=True)
        logits = logits - max
        exp = np.exp(logits)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

    def backward(self, logits, y):
        probability = self.forward(logits)
        return (probability - y) / logits.shape[0]

    def loss(self, logits, y, eps=1e-05):
        probability = self.forward(logits)
        return - np.mean(np.sum(y * np.log(probability + eps), axis=1), axis=0, keepdims=True)

