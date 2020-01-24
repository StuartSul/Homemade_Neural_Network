import random

class Node:
    def __init__(self, id, input_count, activation):
        self.id = id
        self.input_count = input_count
        self.activation = activation
        self.weights = []
        for i in range(self.input_count + 1):
            self.weights.append(2 * random.random() - 1)

    def execute(self, inputs):
        output = self.weights[0]
        for i in range(self.input_count):
            output += inputs[i] * self.weights[i + 1]
        output = self.activation(output)
        return output

    def reset(self):
        for i in range(self.input_count + 1):
            self.weights[i] = (2 * random() - 1)