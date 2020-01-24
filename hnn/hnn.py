from hnn.network import Network
from hnn.trainer import Trainer
from hnn.loader import *

class hnn:
    def __init__(self, network_id, input_count, output_count, structure, activation, 
                    features, labels, train_ratio, loss_function, regularization=None):
        self.network = None
        self.trainer = None
        self.init_network(network_id, input_count, output_count, structure, activation)
        self.init_trainer(features, labels, train_ratio, loss_function, regularization)

    def init_network(self, network_id, input_count, output_count, structure, activation):
        if type(network_id) is not str:
            print('ERROR: must provide string value for network_id')
            return
        if type(input_count) is not int or input_count < 1:
            print('ERROR: input_count must be an integer value greater than or equal to 1')
            return
        elif type(output_count) is not int or output_count < 1:
            print('ERROR: output_count must be an integer value greater than or equal to 1')
            return
        elif type(structure) is not list or type(structure[0]) is not int:
            print('ERROR: structure must be a list of integers')
            return

        self.network = Network(network_id, input_count, output_count, structure, activation)

    def init_trainer(self, features, labels, train_ratio, loss_function, regularization):
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
        elif len(features[0]) != self.network.input_count:
            print('ERROR: input_width and the length of each feature must be equal')
            return
        elif type(labels[0]) is not int and type(labels[0]) is not float:
            print('ERROR: each label must consist of a single numeric value')
            return

        self.trainer = Trainer(self.network, features, labels, train_ratio, loss_function)

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
            print('Period ' + str(i+1) + ' out of ' + str(periods))
            train_loss, test_loss = self.trainer.train(batch_size, learning_rate, epochs_per_period)
            print('  Training loss: ' + str(train_loss))
            print('  Testing  loss: ' + str(test_loss))
        print('\nTraining complete\n')

    def predict(self, feature):
        if self.network == None:
            print('ERROR: Must initialize network first')
        return self.network.execute(feature)[0]

    def save(self, filename):
        if self.network == None:
            print('ERROR: There does not exist a network to save')
        save_network(self.network, filename)
        print('Successfully saved network ' + self.network.id)

    @classmethod
    def load(cls, filename):
        network = load_network(filename)
        new_hnn = cls('', 1, 1, [1], None, [[1]], [1], .1, None)
        new_hnn.network = network
        new_hnn.trainer = None
        print('Successfully loaded network ' + network.id)
        return new_hnn

    def reset(self):
        self.network = None
        self.trainer = None