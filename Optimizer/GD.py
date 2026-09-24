import numpy as np

class Optimizer():
    def __init__(self):
        self.model = None
        self.loss_func = None
        self.data_loader = None

    def train(self, model, loss_func, data_loader, lr=0.05, epochs=100):
        for epoch in range(epochs):
            loss_epoch = 0

            for X_train, y_train in data_loader:
                logits = model.forward(X_train)
                loss = loss_func.loss(logits, y_train)
                loss_epoch += loss
                grads = loss_func.backward(logits, y_train)
                model.backward(grads)
                model.update(lr)

            print(f"Epoch {epoch + 1}, Loss: {loss_epoch/len(data_loader)}")

    def test(self, X, y, model, loss_func):
        output = model.forward(X)
        softmax = loss_func.softmax(output)

        pred = np.argmax(softmax, axis=1)
        true = np.argmax(y, axis=1)

        correct = pred == true
        accuracy = np.mean(correct)
        print(f"Acurracy: {accuracy}%")

        return accuracy