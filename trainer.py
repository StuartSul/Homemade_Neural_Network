from random import random, sample

import network

class Trainer:
    modification_ratio = 0.1
    loss_modifier = 50

    def __init__(self, network, features, labels, batch_size=5, train_ratio=0.8):
        self.network = network
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def train(self, step_size, reps):
        data_size = len(self.features)
        train_size = int(data_size * self.train_ratio)
        data_set = [(self.features[i], self.labels[i]) for i in range(0, data_size)]
        train_set = []
        test_set = []

        for i in range(train_size):
            index = int(len(data_set) * random())
            train_set.append(data_set[index])
            del data_set[index]
        test_set = list(data_set)
        
        for i in range(0, reps):
            batch = sample(train_set, self.batch_size)
            loss = self.run_batch(batch)
            
            print("\n\nrep " + str(i) + "\nloss: " + str(loss))

            network_nodes = self.network.nodes
            selection_size = int(len(network_nodes) * self.modification_ratio + 1)
            selected_nodes = sample(network_nodes, selection_size)
            self.modify_nodes(selected_nodes, -step_size)
            
            new_loss = self.run_batch(batch)
            if new_loss > loss:
                self.modify_nodes(selected_nodes, 2 * step_size)
                print('increased')
            else:
                print('decreased')
        
        test_loss = self.run_batch(test_set)
        print('\nloss on test set: ' + str(test_loss))
    
    def run_batch(self, batch):
        loss = 0
        for example in batch:
            result = self.network.execute(example[0])
            loss += Trainer.get_loss(result, example[1])
        return loss
    
    @staticmethod
    def get_loss(result, label):
        if label == 'H':
            if result < -0.1:
                return -result * Trainer.loss_modifier
            elif result > 0:
                return -result
            else:
                return 0.1 * Trainer.loss_modifier
        else:
            if result > 0.1:
                return result * Trainer.loss_modifier
            elif result < 0:
                return -result
            else:
                return 0.1 * Trainer.loss_modifier
    
    @staticmethod
    def modify_nodes(nodes, amount):
        for node in nodes:
            for i in range(len(node.weights)):
                node.weights[i] += amount