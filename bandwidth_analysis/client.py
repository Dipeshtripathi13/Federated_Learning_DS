import numpy as np
from model import Model

def train_local_model(client_data, model):
    num_epochs = 10
    learning_rate = 0.1

    for epoch in range(num_epochs):
        inputs, labels = client_data  
        inputs = inputs.reshape(1, -1)  
        outputs = model.forward(inputs)
        loss = np.mean((outputs - labels) ** 2)
        grad = 2 * np.dot(inputs.T, outputs - labels) / inputs.shape[1]  
        model.update(grad, learning_rate)

    return model, loss

def calculate_accuracy(model, client_data):
    inputs, labels = client_data
    predictions = model.predict(inputs.reshape(1, -1))
    accuracy = np.mean((predictions == labels).astype(int))
    return accuracy
