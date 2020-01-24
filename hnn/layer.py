from hnn.node import Node

class Layer:
    def __init__(self, layer_id, input_count, node_count, activation):
        self.id = layer_id
        self.input_count = input_count
        self.node_count = node_count
        self.activation = activation
        self.nodes = []
        for i in range(self.node_count):
            self.nodes.append(Node(self.id + "N" + str(i), self.input_count, self.activation))

    def execute(self, inputs):
        output = []
        for node in self.nodes:
            output.append(node.execute(inputs))
        return output

    def reset(self):
        for node in self.nodes:
            node.reset()