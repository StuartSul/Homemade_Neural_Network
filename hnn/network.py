from random import random
from hnn.layer import Layer

class Network:
    def __init__(self, network_id, input_count, output_count, structure, activation):
        self.id = network_id
        self.input_count = input_count
        self.output_count = output_count
        self.structure = structure
        self.activation = activation

        self.layers = []
        self.nodes = []

        network_builder = [self.input_count] + structure + [self.output_count]
        for i in range(len(network_builder) - 1):
            self.layers.append(Layer('L' + str(i), network_builder[i], network_builder[i + 1], self.activation))

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
        desc = 'Network ID: {} ({} layers with total of {} nodes, {} activation)\n'.format(
                            self.id, len(self.layers), len(self.nodes), self.activation.__doc__)
        for layer in self.layers:
            desc += '    Layer ID: {} (receives {} inputs with {} nodes)\n'.format(
                                layer.id, layer.input_count, len(layer.nodes)
            for node in layer.nodes:
                desc += '        Node ID: {}\n'.format(node.id)
                for i in range(len(node.weights)):
                    desc += '            Weight {}: {}\n'.format(i, nodes.weights[i])
        return desc

    def modify(self, amount, shuffle=False):
        if shuffle == True or self.mod_selection == None:
            self.mod_selection = [int(len(self.nodes) * random())]
            self.mod_selection.append(int(len(self.nodes[self.mod_selection[0]].weights) * random()))
        self.nodes[self.mod_selection[0]].weights[self.mod_selection[1]] += amount