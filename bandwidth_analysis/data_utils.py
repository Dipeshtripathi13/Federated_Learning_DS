from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  # Example dataset, change it to your dataset

def load_data():
    # Load your dataset here (e.g., MNIST, CIFAR-10, etc.)
    data = load_iris()  # Using Iris as an example, you should load your desired dataset
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    return {
        'train': (X_train, y_train),
        'test': (X_test, y_test)
    }

def create_client_datasets(global_data, num_clients):
    # Split the training data into 5 parts (one for each client)
    X_train, y_train = global_data['train']
    client_data = []
    
    # Split data into num_clients parts
    for i in range(num_clients):
        # Split into equal parts
        start = i * len(X_train) // num_clients
        end = (i + 1) * len(X_train) // num_clients
        client_data.append((X_train[start:end], y_train[start:end]))
    
    return client_data
