import numpy as np

def generate_dummy_data(num_clients=3, num_samples=10):
    client_data = []
    for _ in range(num_clients):
        inputs = np.random.randn(num_samples, 10)  # Random inputs
        labels = np.random.randn(num_samples, 1)   # Random labels
        client_data.append((inputs, labels))
    return client_data
