from hnn.node import Node

class Layer:
    def __init__(self, layer_id, input_count, node_count, activation):
        self.id = layer_id
        self.input_count = input_count
        self.node_count = node_count
        self.activation = activation
        self.nodes = []
        for i in range(self.node_count):
            self.nodes.append(Node(self.id + 'N' + str(i), self.input_count + 1, self.activation))
        self.input = None
        self.output = None

    def execute(self, inputs):
        self.input = inputs + [1] # bias
        output = []
        for node in self.nodes:
            output.append(node.execute(self.input))
        self.output = output
        return self.output

    def reset(self):
        self.input = None
        self.output = None
        for node in self.nodes:
            node.reset()