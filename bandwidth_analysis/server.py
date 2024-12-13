import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import Model
from client import train_local_model, calculate_accuracy

def plot_confusion_matrix(global_model, test_loader, device, round_num):
    global_model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predictions = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Round {round_num}")
    plt.show()

def federated_averaging(global_model, client_data_loaders, test_loader, num_rounds, device):
    global_model.to(device)
    print(f"Global model size: {sum(p.numel() for p in global_model.parameters()) * 4 / (1024 ** 2):.2f} MB\n")  # Model size in MB

    for round in range(num_rounds):
        print(f"Starting Round {round + 1}")
        
        # Distribute global model to clients
        client_models = []
        client_accuracies = []
        weight_sizes = []

        for client_loader in client_data_loaders:
            # Initialize local model with global weights
            client_model = Model()
            client_model.load_state_dict(global_model.state_dict())
            
            # Train client model locally
            trained_model = train_local_model(client_loader, client_model, device)
            client_models.append(trained_model)

            # Evaluate client model accuracy
            accuracy = calculate_accuracy(trained_model, client_loader, device)
            client_accuracies.append(accuracy)

            # Calculate weight size shared by the client
            weight_size = sum(p.numel() for p in client_model.parameters()) * 4 / (1024 ** 2)  # Size in MB
            weight_sizes.append(weight_size)

        # Aggregate client models to update global model
        global_state = global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client.state_dict()[key] for client in client_models]).mean(dim=0)
        global_model.load_state_dict(global_state)

        # Evaluate global accuracy
        global_accuracy = calculate_accuracy(global_model, test_loader, device)

        # Print results for the round
        print(f"Round {round + 1} - Global Accuracy: {global_accuracy:.4f}")
        for i, (accuracy, size) in enumerate(zip(client_accuracies, weight_sizes)):
            print(f"Client {i + 1} - Local Accuracy: {accuracy:.4f}, Weight Size Shared: {size:.2f} MB")
        print()

        # Plot confusion matrix
        plot_confusion_matrix(global_model, test_loader, device, round_num=round + 1)
