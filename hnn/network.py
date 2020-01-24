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
        
        self.input = None
        self.output = None

    def execute(self, inputs):
        self.input = inputs
        output = self.input
        for layer in self.layers:
            output = layer.execute(output)
        self.output = output
        return self.output

    def reset(self):
        self.input = None
        self.output = None
        for layer in self.layers:
            layer.reset()

    def __repr__(self):
        desc = 'Network ID: {} ({} layers with total of {} nodes, {} activation)\n'.format(
                            self.id, len(self.layers), len(self.nodes), self.activation.__doc__)
        for layer in self.layers:
            desc += '    Layer ID: {} (receives {} inputs with {} nodes)\n'.format(
                                layer.id, layer.input_count, len(layer.nodes))
            for node in layer.nodes:
                desc += '        Node ID: {}\n'.format(node.id)
                for i in range(len(node.weights)):
                    desc += '            Weight {}: {}\n'.format(i, nodes.weights[i])
        return desc