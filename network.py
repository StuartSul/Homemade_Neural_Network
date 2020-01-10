from random import random

class Network:
    def __init__(self, id, hidden_layer, node_per_layer, num_input):
        self.id = id
        self.hidden_layer = hidden_layer
        self.node_per_layer = node_per_layer
        self.num_input = num_input
        self.input = []
        self.layers = []
        self.layers.append(Layer("L0", self.node_per_layer, self.num_input))
        for i in range(1, hidden_layer):
            self.layers.append(Layer("L" + str(i), self.node_per_layer, self.node_per_layer))
        self.output_layer = Layer("LOUT", 1, self.node_per_layer)
        self.output = None
    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.num_input:
            print("Wrong input given to network " + self.id)
            self.reset()
            return
        self.input = input
        next_input = self.layers[0].execute(self.input)
        for i in range(1, self.hidden_layer):
            next_input = self.layers[i].execute(next_input)
        self.output = self.output_layer.execute(next_input)[0]
        return self.output
    def reset(self):
        self.input = []
        for layer in self.layers:
            layer.reset()
        self.output_layer.reset()
        self.output = None

class Layer:
    def __init__(self, id, node_per_layer, num_input):
        self.id = id
        self.node_per_layer = node_per_layer
        self.num_input = num_input
        self.nodes = []
        for i in range(self.node_per_layer):
            self.nodes.append(Node(self.id + "N" + str(i), self.num_input))
        self.input = []
        self.output = []
    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.num_input:
            print("Wrong input given to layer " + self.id)
            self.reset()
            return
        self.input = input
        self.output = []
        for node in self.nodes:
            self.output.append(node.execute(self.input))
        return self.output
    def reset(self):
        self.input = []
        self.output = []
        for node in self.nodes:
            node.reset()

class Node:
    def __init__(self, id, num_input):
        self.id = id
        self.num_input = num_input
        self.weights = []
        for i in range(num_input + 1):
            self.weights.append(num_input * random())
        self.input = []
        self.output = None
    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.num_input:
            print("Wrong input given to node " + self.id)
            self.reset()
            return
        self.input = input
        output = self.weights[0]
        for i in range(self.num_input):
            output += self.input[i] * self.weights[i + 1]
        self.output = output
        return self.output
    def reset(self):
        self.input = []
        self.output = None