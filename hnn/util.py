import math

class Linear:
    'linear'
    @staticmethod
    def calculate(x):
        return x
    @staticmethod
    def derivative(x):
        return 1

class Sigmoid:
    'sigmoid'
    @staticmethod
    def calculate(x):
        return 1 / (1 + math.exp(-x))
    @staticmethod
    def derivative(x):
        return Sigmoid.activate(x) * (1 - Sigmoid.activate(x))

class Tanh:
    'tanh'
    @staticmethod
    def calculate(x):
        return math.tanh(x)
    @staticmethod
    def derivative(x):
        return 1 - Tanh.activate(x) ** 2

class ReLU:
    'ReLU'
    @staticmethod
    def calculate(x):
        return max([0, x])
    @staticmethod
    def derivative(x):
        return int(x >= 0)

class RMSE:
    def calculate(predictions, labels):
        loss = 0.0
        for i in range(len(predictions)):
            loss += (predictions[i] - labels[i]) * (predictions[i] - labels[i])
        loss = (loss / len(predictions))**0.5
        return loss
    def derivative(prediction, label):
        return 2 * (prediction - label)

class LogLoss:
    def calculate(predictions, labels):
        loss = 0.0
        for i in range(len(predictions)):
            loss += (labels[i] - 1) * math.log(1 - predictions[i]) - labels[i] * math.log(predictions[i])
        loss /= len(predictions)
        return loss
    def derivative(prediction, label):
        return (label - prediction) / (prediction * (prediction - 1))

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