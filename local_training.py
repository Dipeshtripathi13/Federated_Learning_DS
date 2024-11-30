import data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Local training function
def train_local_model(data):
    X = data[['Age', 'BMI', 'BloodPressure']]
    y = data['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, acc, X_test, y_test

# Train local models for each client
local_models = []
local_accuracies = []
test_data = []
client_data = data.client_data
for client in client_data:
    model, acc, X_test, y_test = train_local_model(client)
    local_models.append(model)
    local_accuracies.append(acc)
    test_data.append((X_test, y_test))

print("Local Model Accuracies:", local_accuracies)
