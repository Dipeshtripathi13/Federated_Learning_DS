import torch
import torch.nn as nn
import torch.optim as optim

#normal and fedavg
# def train_local_model(client_data, model, device):
#     model.to(device)
#     model.train()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(5):  # Local epochs
#         for images, labels in client_data:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#     return model

#fedprox

def train_local_model(client_loader, model, global_weights, device, mu=0.1, num_epochs=5):
    model.train()
    model.to(device)
    global_weights = {k: v.to(device) for k, v in global_weights.items()}  # Move global weights to device
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for images, labels in client_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Compute standard loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add proximal term to the loss
            prox_term = 0
            for param_name, param in model.named_parameters():
                prox_term += torch.sum((param - global_weights[param_name]) ** 2)
            loss += (mu / 2) * prox_term
            
            loss.backward()
            optimizer.step()

    return model

def calculate_accuracy(model, data_loader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total
