import numpy as np
from model import Model
from client import train_local_model, calculate_accuracy

def federated_averaging(global_model, client_data, num_rounds):
    for round in range(num_rounds):
        print(f"Starting Round {round + 1}")
        
        # Client selection
        selected_clients = np.random.choice(range(len(client_data)), size=3, replace=False)  # Select 3 clients
        
        # Local training
        client_models = []
        client_accuracies = []
        local_losses = []
        for i, client_index in enumerate(selected_clients):
            client_data_ = client_data[client_index]
            client_model = Model()
            local_model, loss = train_local_model(client_data_, client_model)
            client_models.append(local_model)
            local_accuracy = calculate_accuracy(local_model, client_data_)
            client_accuracies.append(local_accuracy)
            local_losses.append(loss)

        # Model aggregation
        aggregated_model = Model()
        for client_model in client_models:
            aggregated_model.fc += client_model.fc

        # Model averaging
        global_model.fc = aggregated_model.fc / len(client_models)

        # Calculate global accuracy
        global_accuracy = calculate_accuracy(global_model, client_data[selected_clients[0]])

        # Print model parameters and accuracies
        print(f"Round {round+1} - Global Model Parameters:")
        print(global_model.fc)
        print(f"Round {round+1} - Global Accuracy: {global_accuracy:.4f}")
        for i, client_accuracy in enumerate(client_accuracies):
            print(f"Client {selected_clients[i] + 1} Local Accuracy: {client_accuracy:.4f}")
        print("Model Update Sizes (in bytes):")
        for i, client_model in enumerate(client_models):
            model_update_size = client_model.fc.nbytes  # Size in bytes of the model update
            print(f"Client {selected_clients[i] + 1} Model Update Size: {model_update_size} bytes")
        print()
