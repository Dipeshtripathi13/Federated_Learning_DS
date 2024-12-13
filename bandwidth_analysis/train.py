import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import Model
from server import federated_averaging

# Hyperparameters
num_clients = 5
num_rounds = 5
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split dataset among clients
client_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients)
client_data_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize global model
global_model = Model()

# Train Federated Learning
federated_averaging(global_model, client_data_loaders, test_loader, num_rounds, device)
