import operator
import random

class Node:
    def __init__(self, node_id, input_count, activation):
        self.id = node_id
        self.input_count = input_count
        self.activation = activation
        self.weights = []
        for i in range(self.input_count):
            self.weights.append(2 * random.random() - 1)

    def execute(self, inputs):
        output = map(operator.mul, inputs, self.weights)
        output = sum(output)
        output = self.activation.calculate(output)
        return output

    def reset(self):
        for i in range(self.input_count):
            self.weights[i] = (2 * random() - 1)