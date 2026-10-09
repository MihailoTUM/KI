import numpy as np
import pandas as pd
import kagglehub

class DataLoader():
    def __init__(self, X, y, batch=32):
        self.index = 0
        self.batch = batch
        self.X = X
        self.y = y

    def __iter__(self):
        self.index = 0
        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation]
        self.y = self.y[permutation]
        return self

    def __next__(self):
        if self.index + self.batch >= self.X.shape[0]:
            raise StopIteration

        end = self.index + self.batch
        p = (self.X[self.index:end], self.y[self.index:end])
        self.index = end

        return p

    def __len__(self):
        return self.X.shape[0] // self.batch


