import numpy as np

class FFN():
    def __init__(self, input_n, hidden_n, output_n):
        self.w_1 = np.random.rand(input_n, hidden_n) - 0.5
        self.b_1 = np.random.rand(hidden_n) - 0.5

        self.w_2 = np.random.rand(hidden_n, output_n) - 0.5
        self.b_2 = np.random.rand(output_n) - 0.5

        self.w_1_grads = np.zeros_like(self.w_1)
        self.b_1_grads = np.zeros_like(self.b_1)

        self.w_2_grads = np.zeros_like(self.w_2)
        self.b_2_grads = np.zeros_like(self.b_2)

    def relu(self, X):
        return np.maximum(0, X)

    def reluDeriv(self, X):
        return (X > 0).astype(float)

    def forward(self, X):
        self.input = X
        out = X @ self.w_1 + self.b_1
        self.z1 = out
        out = self.relu(out)
        self.a1 = out
        out = out @ self.w_2 + self.b_2
        self.z2 = out

        return out

    def backward(self, grads):
        self.w_2_grads = self.a1.T @ grads
        self.b_2_grads = np.mean(grads, axis=0, keepdims=False)

        """
            grads = (exp, output)
            a1 = (exp, hidden)
            w2 = (hidden, output)
            b2 = (output)
            input = (exp, input)
            w1 = (input, hidden)
        """
        dL_da1 = grads @ self.w_2.T
        dL_dz1 = dL_da1 * self.reluDeriv(self.z1)

        self.w_1_grads = self.input.T @ dL_dz1
        self.b_1_grads = np.mean(dL_dz1, axis=0, keepdims=False)

    def update(self, lr=0.1):
        self.w_1 -= lr * self.w_1_grads
        self.b_1 -= lr * self.b_1_grads

        self.w_2 -= lr * self.w_2_grads
        self.b_2 -= lr * self.b_2_grads
