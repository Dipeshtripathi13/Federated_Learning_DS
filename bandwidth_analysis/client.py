import numpy as np
from sklearn.linear_model import LogisticRegression

class FederatedClient:
    def __init__(self, data):
        self.X_train, self.y_train = data  # Unpack the dataset
        self.model = LogisticRegression(max_iter=1000)  # Model initialization

    def train(self):
        # Train the model locally
        self.model.fit(self.X_train, self.y_train)
        # Return model coefficients (weights)
        return self.model.coef_.flatten()

    def evaluate(self):
        # Evaluate the local model accuracy on its data
        return np.mean(self.model.predict(self.X_train) == self.y_train)
