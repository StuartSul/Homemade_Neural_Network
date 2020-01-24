from hnn.network import Network
from hnn.util import *

magic = 'hnn_network70527919'
separator = '\n'

def load_network(filename):
    with open(filename, 'r') as file:
        data = file.read()
        if data.startswith(magic):
            data = data.split(separator)
        else:
            print('ERROR: Wrong file format')
            return None

    network_id = data[1]
    input_count = int(data[2])
    output_count = int(data[3])
    structure = []

    index = 5
    for i in range(int(data[4])):
        structure.append(int(data[index]))
        index += 1
    
    activation_doc = data[index]
    index += 1

    if activation_doc == 'linear':
        activation = Linear()
    elif activation_doc == 'sigmoid':
        activation = Sigmoid()
    elif activation_doc == 'tanh':
        activation = Tanh()
    elif activation_doc == 'ReLU':
        activation = ReLU()

    network = Network(network_id, input_count, output_count, structure, activation)

    for node in network.nodes:
        for i in range(len(node.weights)):
            node.weights[i] = float(data[index])
            index += 1
    
    return network

def save_network(network, filename):
    data = magic + separator
    data += network.id + separator +\
            str(network.input_count) + separator +\
            str(network.output_count) + separator +\
            str(len(network.structure))
    
    for node_count in structure:
        data += separator + str(node_count)
    
    data += separator + network.activation.__doc__

    for node in network.nodes:
        for weight in node.weights:
            data += separator + str(weight)
    
    with open(filename, 'w') as file:
        file.write(data)