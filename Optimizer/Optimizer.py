import numpy as np
import torch

class Optimizer():
    def __init__(self):
        pass

    def train(self, model, loss_func, data_loader, lr=0.05, epochs=100, momentum=0, lr_decay=0.1, n=10):
        pass

    def test(self, X, y, model, loss_func):
        pass

    def schedule(self, lr, lr_decay, epoch, n):
        pass