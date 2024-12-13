from model import Model
from server import federated_averaging
import numpy as np

# Example usage

global_model = Model()
client_data = [
    (np.random.randn(10), np.random.randn(1)),
    (np.random.randn(10), np.random.randn(1)),
    (np.random.randn(10), np.random.randn(1)),
    (np.random.randn(10), np.random.randn(1)),
    (np.random.randn(10), np.random.randn(1)),
]  # Dummy client data
num_rounds = 2
federated_averaging(global_model, client_data, num_rounds)
