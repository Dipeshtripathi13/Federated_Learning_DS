import numpy as np

def fed_avg(client_weights):
    # Ensure weights are consistent in shape
    processed_weights = [np.squeeze(np.array(w)) for w in client_weights]
    return np.mean(processed_weights, axis=0)



def fed_prox(client_weights, global_weights, mu=0.1):
    prox_weights = []
    for cw in client_weights:
        prox_weights.append((1 - mu) * cw + mu * global_weights)
    return np.mean(prox_weights, axis=0)

def sparsify_weights(weights, threshold=0.01):
    return np.where(np.abs(weights) > threshold, weights, 0)
