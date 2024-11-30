import aggrigate
import local_training
from sklearn.metrics import accuracy_score
global_accuracies = []
test_data = local_training.test_data
aggregated_weights = aggrigate.aggregated_weights
for X_test, y_test in test_data:
    predictions = (X_test @ aggregated_weights.T).flatten()
    predictions = (predictions > 0.5).astype(int)
    acc = accuracy_score(y_test, predictions)
    global_accuracies.append(acc)

print("Global Model Accuracies on Test Sets:", global_accuracies)
