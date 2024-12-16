import time  # Import time module for tracking training duration
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import Model
from client import train_local_model, calculate_accuracy


def federated_prox(global_model, client_data_loaders, test_loader, num_rounds, device, mu=0.1):
    global_model.to(device)
    print(f"Global model size: {sum(p.numel() for p in global_model.parameters()) * 4 / (1024 ** 2):.2f} MB\n")  # Model size in MB

    for round in range(num_rounds):
        start_time = time.time()  # Start time for the round
        print(f"Starting Round {round + 1}")
        
        client_models = []
        client_accuracies = []
        weight_sizes = []
        local_data_sizes = []

        # Get global weights
        global_weights = global_model.state_dict()

        for client_loader in client_data_loaders:
            # Initialize local model with global weights
            client_model = Model()
            client_model.load_state_dict(global_weights)
            
            # Train client model locally with proximal regularization
            trained_model = train_local_model(client_loader, client_model, global_weights, device, mu=mu)
            client_models.append(trained_model)

            # Evaluate client model accuracy
            accuracy = calculate_accuracy(trained_model, client_loader, device)
            client_accuracies.append(accuracy)

            # Calculate weight size shared by the client
            weight_size = sum(p.numel() for p in client_model.parameters()) * 4 / (1024 ** 2)  # Size in MB
            weight_sizes.append(weight_size)

            # Store the local data size for weighted aggregation
            local_data_sizes.append(len(client_loader.dataset))

        # Weighted aggregation for FedProx
        global_state = global_model.state_dict()
        total_data_points = sum(local_data_sizes)

        for key in global_state.keys():
            global_state[key] = torch.stack(
                [client.state_dict()[key] * (local_data_sizes[i] / total_data_points) for i, client in enumerate(client_models)]
            ).sum(dim=0)
        global_model.load_state_dict(global_state)

        # Evaluate global accuracy
        global_accuracy = calculate_accuracy(global_model, test_loader, device)

        # End time for the round
        round_time = time.time() - start_time

        # Print results for the round
        print(f"Round {round + 1} - Training Time: {round_time:.2f} seconds")
        print(f"Round {round + 1} - Global Accuracy: {global_accuracy:.4f}")
        for i, (accuracy, size) in enumerate(zip(client_accuracies, weight_sizes)):
            print(f"Client {i + 1} - Local Accuracy: {accuracy:.4f}, Weight Size Shared: {size:.2f} MB")
        print()
