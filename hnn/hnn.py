from .network import Network
from .trainer import Trainer
from .loader import *


class hnn:
    def __init__(self, network_name, input_width, num_hidden_layers, 
                    node_per_layer, features, labels, train_ratio):
        self.network = None
        self.trainer = None
        self.init_network(network_name, input_width, num_hidden_layers, node_per_layer)
        self.init_trainer(features, labels, train_ratio)

    def init_network(self, network_name, input_width, 
                        num_hidden_layers, node_per_layer):
        if type(network_name) is not str:
            print('ERROR: must provide string value for network_name')
            return
        if type(input_width) is not int or input_width < 1:
            print('ERROR: input_width must be an integer value greater than or equal to 1')
            return
        elif type(num_hidden_layers) is not int or num_hidden_layers < 1:
            print('ERROR: num_hidden_layers must be an integer value greater than or equal to 1')
            return
        elif type(node_per_layer) is not int or num_hidden_layers < 1:
            print('ERROR: node_per_layer must be an integer value greater than or equal to 1')
            return

        self.network = Network(network_name, input_width, num_hidden_layers, node_per_layer)

    def init_trainer(self, features, labels, train_ratio):
        if self.network == None:
            print('ERROR: must initialize network first')
            return
        elif type(train_ratio) is not float or train_ratio >= 1 or train_ratio <= 0:
            print('ERROR: train_ratio must be a float value at the range of (0, 1)')
            return
        elif type(features) is not list:
            print('ERROR: must provide an instance of list for features')
            return
        elif type(labels) is not list:
            print('ERROR: must provide an instance of list for labels')
            return
        elif len(features) == 0 or len(labels) == 0 or len(features) != len(labels):
            print('ERROR: features and labels must have same length')
            return
        elif len(features[0]) != self.network.input_width:
            print('ERROR: input_width and the length of each feature must be equal')
            return
        elif type(labels[0]) is not int and type(labels[0]) is not float:
            print('ERROR: each label must consist of a single numeric value')
            print(labels[0])
            return

        self.trainer = Trainer(self.network, features, labels, train_ratio)

    def train(self, batch_size, learning_rate, total_epochs, periods):
        if self.network == None:
            print('ERROR: must initialize network first')
            return
        elif self.trainer == None:
            print('ERROR: must initialize trainer first')
            return
        if type(batch_size) is not int or batch_size < 1 or batch_size > len(self.network.nodes):
            print('ERROR: batch_size must be an integer value at the range of [1, NUM_NODES]')
            return
        elif type(learning_rate) is not float or learning_rate <= 0:
            print('ERROR: learning_rate must be a float value greater than 0')
            return
        elif type(total_epochs) is not int or total_epochs < 1:
            print('ERROR: total_epochs must be an integer value greater than 0')
            return
        elif type(periods) is not int or periods < 1 or periods > total_epochs:
            print('ERROR: periods must be an integer value at the range of [1, total_epochs]')
            return
            
        epochs_per_period = total_epochs // periods
        print('\nInitiating training on network ' + self.network.id + "...\n")
        for i in range(periods):
            print("Period " + str(i+1) + " out of " + str(periods))
            print("  loss: " + str(self.trainer.train(batch_size, learning_rate, epochs_per_period)))
        print('\nTraining complete\n')

    def predict(self, feature):
        if self.network == None:
            print('ERROR: Must initialize network first')
        return self.network.execute(feature)

    def save(self, filename):
        if self.network == None:
            print('ERROR: There does not exist a network to save')
        save_network(self.network, filename)
        print('Successfully saved network ' + self.network.id)

    @classmethod
    def load(cls, filename):
        network = load_network(filename)
        new_hnn = cls('', 1, 1, 1, [[1]], [1], .1)
        new_hnn.network = network
        #new_hnn.trainer = None
        print('Successfully loaded network ' + network.id)
        return new_hnn

    def reset(self):
        self.network = None
        self.trainer = None