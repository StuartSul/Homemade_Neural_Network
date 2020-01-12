from .network import Network

magic = 'sffjvi'
separator = 'hnnssfjviha*(!#!)@'

def load_network(filename):
    with open(filename, 'r') as file:
        data = file.read()
        if data.startswith(magic):
            data = data.split(separator)
        else:
            print('ERROR: Wrong file format')
            return None

    network = Network(data[1], int(data[2]),
                        int(data[3]), int(data[4]))

    index = 5
    for node in network.nodes:
        for i in range(len(node.weights)):
            node.weights[i] = float(data[index])
            index += 1
    
    return network

def save_network(network, filename):
    data = magic + separator
    data += network.id + separator +\
            str(network.input_width) + separator +\
            str(network.num_hidden_layer) + separator +\
            str(network.node_per_layer)

    for node in network.nodes:
        for weight in node.weights:
            data += separator + str(weight)
    
    with open(filename, 'w') as file:
        file.write(data)