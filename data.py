# Simulating non-IID healthcare data
import numpy as np
import pandas as pd


# Generate synthetic data
np.random.seed(42)
n_clients = 5
n_samples = 1000

# Features: Age, BMI, Blood Pressure; Target: Risk (0 or 1)
data = pd.DataFrame({
    'Age': np.random.randint(20, 70, n_samples),
    'BMI': np.random.uniform(18, 35, n_samples),
    'BloodPressure': np.random.randint(80, 140, n_samples),
    'Risk': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
})

# Split data non-IID across clients
client_data = []
for i in range(n_clients):
    client_data.append(data.sample(frac=0.2, replace=False).reset_index(drop=True))
