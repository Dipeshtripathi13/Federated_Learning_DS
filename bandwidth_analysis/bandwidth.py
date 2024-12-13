import sys

def calculate_bandwidth(weights):
    # This function calculates the size of model update in bytes
    weights_size = sys.getsizeof(weights)  # Get size in bytes
    return weights_size
