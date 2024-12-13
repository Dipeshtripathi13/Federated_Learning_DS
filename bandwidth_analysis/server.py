from sklearn.linear_model import LogisticRegression
class FederatedServer:
    def __init__(self):
        # Initialize global model, this could be a Logistic Regression or any other model
        self.model = LogisticRegression(max_iter=1000)

    def set_weights(self, new_weights):
        # Set the global model weights to the new averaged weights
        self.model.coef_ = new_weights.reshape(self.model.coef_.shape)

    def evaluate(self, test_data):
        X_test, y_test = test_data
        return np.mean(self.model.predict(X_test) == y_test)
