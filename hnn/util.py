import math

# Activation Functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x)
    return math.tanh(x)

def ReLU(x)
    return max([0, x])

# Loss Functions
def RMSE(predictions, label):
    loss = 0.0
    for prediction in predictions:
        loss += (prediction - label) * (prediction - label)
    return (loss / len(predictions))**0.5

def log_loss(predictions, label):
    loss = 0.0
    for prediction in predictions:
        loss += (label - 1) * math.log(1 - prediction) - label * math.log(prediction)
    return loss / len(predictions)

# Regularization
def L1(weights, lamb):
    complexity = 0.0
    for weight in weights:
        complexity += abs(weight)
    return lamb * complexity

def L2(weights, lamb):
    complexity = 0.0
    for weight in weights:
        complexity += weight * weight
    return lamb * complexity