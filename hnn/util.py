import math

class Activation:
    @staticmethod
    def linear(x):
        'linear'
        return x

    @staticmethod
    def sigmoid(x):
        'sigmoid'
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def tanh(x):
        'tanh'
        return math.tanh(x)

    @staticmethod
    def ReLU(x):
        'ReLU'
        return max([0, x])

class Loss:
    @staticmethod
    def RMSE(predictions, label):
        loss = 0.0
        for prediction in predictions:
            loss += (prediction - label) * (prediction - label)
        return (loss / len(predictions))**0.5

    @staticmethod
    def log_loss(predictions, label):
        loss = 0.0
        for prediction in predictions:
            loss += (label - 1) * math.log(1 - prediction) - label * math.log(prediction)
        return loss / len(predictions)

class Regularization:
    @staticmethod
    def L1(weights, lamb):
        complexity = 0.0
        for weight in weights:
            complexity += abs(weight)
        return lamb * complexity

    @staticmethod
    def L2(weights, lamb):
        complexity = 0.0
        for weight in weights:
            complexity += weight * weight
        return lamb * complexity