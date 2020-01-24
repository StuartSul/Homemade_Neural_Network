from random import random
from hnn.layer import Layer

class Network:
    def __init__(self, id, input_count, output_count, structure, activation):
        self.id = id
        self.input_count = input_count
        self.output_count = output_count
        self.structure = structure
        self.activation = activation

        self.layers = []
        self.nodes = []

        network_builder = [self.input_count] + structure + [self.output_count]
        for i in range(len(network_builder) - 1):
            self.layers.append(Layer("L" + str(i), network_builder[i], network_builder[i + 1], self.activation))

        for layer in self.layers:
            for node in layer.nodes:
                self.nodes.append(node)

        self.mod_selection = None

    def execute(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.execute(output)
        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()

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