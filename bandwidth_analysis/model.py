import numpy as np

class Model:
    def __init__(self):
        self.fc = np.random.randn(10, 1)
    
    def forward(self, x):
        return np.dot(x, self.fc)

    def predict(self, x):
        return np.dot(x, self.fc)

    def update(self, grad, learning_rate=0.1):
        self.fc -= learning_rate * grad
