import numpy as np
from Loss.Loss import Loss
from DataLoader.DataLoader import DataLoader
from Optimizer.Optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self):
        super().__init__()
        self.model = None
        self.loss_func = None
        self.data_loader = None

    def schedule(self, lr, lr_decay, epoch, n):
        return lr * lr_decay**(epoch // n)

    def train(self, model, loss_func: Loss, data_loader: DataLoader, lr=0.05, epochs=100, momentum=0, lr_decay=0.1, n=10):
        for epoch in range(epochs):
            loss_epoch = 0

            for X_train, y_train in data_loader:
                print("Hello")
                logits = model.forward(X_train, training=True)
                loss = loss_func.loss(logits, y_train)
                print("loss:", loss)
                loss_epoch += loss
                grads = loss_func.backward(logits, y_train)
                model.backward(grads, momentum)
                model.update(self.schedule(lr, lr_decay, epoch, n))
                model.reset()

            print(f"Epoch {epoch + 1}, Loss: {loss_epoch/len(data_loader)}")

    def test(self, X, y, model, loss_func):
        output = model.forward(X, training=False)
        softmax = loss_func.forward(output)

        pred = np.argmax(softmax, axis=1)
        true = np.argmax(y, axis=1)

        correct = pred == true
        accuracy = np.mean(correct)
        print(f"Acurracy: {accuracy*100}%")

        return accuracy