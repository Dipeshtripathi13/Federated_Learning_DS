import local_training
import numpy as np
def personalized_aggregation(local_models, client_data):
    global_weights = np.zeros_like(local_models[0].coef_)
    
    # Aggregate weights based on similarity
    for i, model in enumerate(local_models):
        similarity_weight = 1 / (1 + np.var(client_data[i]['Age']))  # Example: Use variance of a feature
        global_weights += similarity_weight * model.coef_
    
    global_weights /= len(local_models)
    return global_weights
local_models = local_training.local_models
client_data = local_training.client_data
# Update global model
aggregated_weights = personalized_aggregation(local_models, client_data)
