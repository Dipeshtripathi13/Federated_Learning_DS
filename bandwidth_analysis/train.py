import numpy as np
from tqdm import tqdm
from client import FederatedClient
from server import FederatedServer
from approaches import fed_avg
from data_utils import load_data, create_client_datasets
from bandwidth import calculate_bandwidth

# Load data and create client datasets
global_data = load_data()  # Load your dataset (e.g., MNIST or CIFAR-10)
client_datasets = create_client_datasets(global_data, num_clients=5)

# Initialize clients
clients = [FederatedClient(data) for data in client_datasets]

# Initialize global model
global_model = FederatedServer()

# Simulate Federated Learning for 10 rounds
for round in range(10):  # Simulate 10 communication rounds
    print(f"Starting Round {round + 1}")
    
    client_weights = []
    local_accuracies = []
    
    # Training clients
    for i, client in enumerate(tqdm(clients, desc="Training Clients")):
        client_accuracy = client.evaluate()  # Local accuracy before training
        local_accuracies.append(client_accuracy)
        print(f"Client {i + 1} Local Accuracy before training: {client_accuracy:.4f}")
        
        # Train the client model locally
        weights = client.train()  # Train and get model weights
        print(f"Client {i + 1} Local Model Updated")
        client_weights.append(weights)
        
        # Print size of the model update sent to the global model
        update_size = calculate_bandwidth(weights)
        print(f"Client {i + 1} Model Update Size: {update_size} bytes")
    
    # Aggregating weights using FedAvg
    global_model_weights = fed_avg(client_weights)
    global_model.set_weights(global_model_weights)
    
    # Evaluate global model on test set
    global_accuracy = global_model.evaluate(global_data['test'])
    print(f"Round {round + 1} Global Accuracy: {global_accuracy:.4f}")
    print("-" * 50)
