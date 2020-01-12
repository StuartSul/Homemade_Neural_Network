from random import random

class Network:
    def __init__(self, id, input_width, num_hidden_layer, node_per_layer):
        self.id = id
        self.num_hidden_layer = num_hidden_layer
        self.node_per_layer = node_per_layer
        self.input_width = input_width
        self.input = []
        self.layers = []
        self.nodes = []
        self.layers.append(Layer("L0", self.input_width, self.node_per_layer))
        for i in range(1, num_hidden_layer):
            self.layers.append(Layer("L" + str(i), self.node_per_layer, self.node_per_layer))
        self.output_layer = Layer("LOUT", self.node_per_layer, 1)
        self.output = None
        for layer in self.layers:
            for node in layer.nodes:
                self.nodes.append(node)
        self.nodes.append(self.output_layer.nodes[0])
        self.mod_selection = None

    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.input_width:
            print("ERROR: Wrong input given to network " + self.id)
            self.reset()
            return
        self.input = input
        next_input = self.layers[0].execute(self.input)
        for i in range(1, self.num_hidden_layer):
            next_input = self.layers[i].execute(next_input)
        self.output = self.output_layer.execute(next_input)[0]
        return self.output

    def reset(self):
        self.input = []
        for layer in self.layers:
            layer.reset()
        self.output_layer.reset()
        self.output = None

    def __repr__(self):
        desc = "Network ID: " + self.id + '\n'
        desc += "Current input: " + str(self.input) + '\n'
        for layer in self.layers:
            desc += "    Layer ID: " + layer.id + '\n'
            desc += "        Given input: " + str(layer.input) + '\n'
            for node in layer.nodes:
                desc += "        Node ID: " + node.id + '\n'
                desc += "            Weights: " + str(node.weights) + '\n'
                desc += "            Current output: " + str(node.output) + '\n'
        desc += "    Layer ID: " + self.output_layer.id + '\n'
        for node in self.output_layer.nodes:
            desc += "        Node ID: " + node.id + '\n'
            desc += "            Weights: " + str(node.weights) + '\n'
            desc += "            Current output: " + str(node.output) + '\n'
        desc += "Final output: " + str(self.output) + '\n\n'
        return desc

    def modify(self, amount, shuffle=False):
        if shuffle == True or self.mod_selection == None:
            self.mod_selection = [int(len(self.nodes) * random())]
            self.mod_selection.append(int(len(self.nodes[self.mod_selection[0]].weights) * random()))
        self.nodes[self.mod_selection[0]].weights[self.mod_selection[1]] += amount
    

class Layer:
    def __init__(self, id, input_width, node_per_layer):
        self.id = id
        self.node_per_layer = node_per_layer
        self.input_width = input_width
        self.nodes = []
        for i in range(self.node_per_layer):
            self.nodes.append(Node(self.id + "N" + str(i), self.input_width))
        self.input = []
        self.output = []

    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.input_width:
            print("ERROR: Wrong input given to layer " + self.id)
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
    def __init__(self, id, input_width):
        self.id = id
        self.input_width = input_width
        self.weights = []
        for i in range(input_width + 1):
            self.weights.append(random() - 0.5)
        self.input = []
        self.output = None

    def execute(self, input):
        if type(input) != type(self.input) or len(input) != self.input_width:
            print("ERROR: Wrong input given to node " + self.id)
            self.reset()
            return
        self.input = input
        output = self.weights[0]
        for i in range(self.input_width):
            output += self.input[i] * self.weights[i + 1]
        self.output = output
        return self.output

    def reset(self):
        self.input = []
        self.output = None