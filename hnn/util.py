import math

class Linear:
    'linear'
    @staticmethod
    def calculate(x):
        return x
    @staticmethod
    def derivative(y):
        return 1

class Sigmoid:
    'sigmoid'
    @staticmethod
    def calculate(x):
        return 1 / (1 + math.exp(-x))
    @staticmethod
    def derivative(y):
        return y * (1 - y)

class Tanh:
    'tanh'
    @staticmethod
    def calculate(x):
        return math.tanh(x)
    @staticmethod
    def derivative(y):
        return 1 - y * y

class ReLU:
    'ReLU'
    @staticmethod
    def calculate(x):
        return max([0, x])
    @staticmethod
    def derivative(y):
        return int(y > 0)

class RMSE:
    @staticmethod
    def calculate(predictions, labels):
        loss = 0.0
        for i in range(len(predictions)):
            loss += (predictions[i] - labels[i]) * (predictions[i] - labels[i])
        loss = (loss / len(predictions))**0.5
        return loss
    @staticmethod
    def derivative(prediction, label):
        return 2 * (prediction - label)

class LogLoss:
    eps = 0.000000000001
    @staticmethod
    def calculate(predictions, labels):
        loss = 0.0
        for i in range(len(predictions)):
            loss += (labels[i] - 1) * math.log(1 - predictions[i] + LogLoss.eps)
            loss -= labels[i] * math.log(predictions[i] + LogLoss.eps)
        loss /= len(predictions)
        return loss
    @staticmethod
    def derivative(prediction, label):
        return (label - prediction) / (prediction * (prediction - 1) + LogLoss.eps)

class L1:
    @staticmethod
    def calculate(weights, lamb):
        complexity = 0.0
        for weight in weights:
            complexity += abs(weight)
        complexity *= lamb
        return complexity

class L2:
    @staticmethod
    def calculate(weights, lamb):
        complexity = 0.0
        for weight in weights:
            complexity += weight * weight
        complexity *= lamb
        return complexity