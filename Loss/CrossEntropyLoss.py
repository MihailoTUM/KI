import numpy as np
import torch

class CrossEntropyLoss():
    def __init__(self):
        pass

    def softmax(self, logits):
        max = np.max(logits, axis=1, keepdims=True)
        logits = logits - max
        exp = np.exp(logits)
        sum = np.sum(exp, axis=1, keepdims=True)
        return exp / sum

    def backward(self, logits, y):
        probability = self.softmax(logits)
        return probability - y

    def loss(self, logits, y, eps=1e-05):
        probability = self.softmax(logits)
        return - np.mean(np.sum(y * np.log(probability + eps), axis=1), axis=0, keepdims=True)

