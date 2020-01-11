from random import random, sample

import network

class Trainer:
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
        
        batch = sample(train_set, self.batch_size)
        for i in range(0, reps):
            print()
            print("rep " + str(i))
            loss = 0.0
            for example in batch:
                result = self.network.execute(example[0])
                if (result >= 0 and example[1] == 'L') or\
                   (result < 0 and example[1] == 'H'):
                    loss += 1.0
            print("loss: " + str(loss))

            selected_nodes = sample(self.network.nodes, self.network.hidden_layer)
            for node in selected_nodes:
                for i in range(len(node.weights)):
                    node.weights[i] -= step_size
            print('reduced')
            
            new_loss = 0.0
            for example in batch:
                result = self.network.execute(example[0])
                if (result >= 0 and example[1] == 'L') or\
                   (result < 0 and example[1] == 'H'):
                    new_loss += 1.0

            print ('new_loss: ' + str(new_loss))
            if new_loss > loss:
                print('increased')
                for node in selected_nodes:
                    for i in range(len(node.weights)):
                        node.weights[i] += 2 * step_size
        
        loss = 0.0
        for test_example in test_set:
            result = self.network.execute(test_example[0])
            if (result >= 0 and test_example[1] == 'L') or\
                (result < 0 and test_example[1] == 'H'):
                loss += 1.0
        print('loss on test set: ' + str(loss))