from random import random, sample
from math import log

from .network import Network

class Trainer:
    def __init__(self, network, features, labels, train_ratio):
        self.network = network
        self.features = features
        self.labels = labels
        self.train_set, self.test_set = self.divide_data(train_ratio)

    def train(self, batch_size, learning_rate, epochs):
        for i in range(0, epochs):
            batch = sample(self.train_set, batch_size)

            original_loss = self.run_batch(batch)
            self.network.modify(-learning_rate, shuffle=True)
            down_loss = self.run_batch(batch)
            self.network.modify(2 * learning_rate)
            up_loss = self.run_batch(batch)
            
            if down_loss >= original_loss and up_loss >= original_loss:
                self.network.modify(-learning_rate)
            elif down_loss < up_loss:
                self.network.modify(-2 * learning_rate)
                
        test_loss = self.run_batch(self.test_set)
        return test_loss

    def shuffle_data(self, train_ratio=None):
        if train_ratio == None:
            train_ratio = self.train_ratio
        self.train_set, self.test_set = self.divide_data(train_ratio)
    
    def divide_data(self, train_ratio):
        size = len(self.features)
        train_size = int(size * train_ratio)

        data_set = [(self.features[i], self.labels[i]) for i in range(0, size)]
        train_set, test_set = [], []

        for i in range(train_size):
            index = int(len(data_set) * random())
            train_set.append(data_set[index])
            del data_set[index]
        test_set = list(data_set)

        return train_set, test_set
    
    def run_batch(self, batch):
        loss = 0
        for example in batch:
            result = self.network.execute(example[0])
            loss += abs(result - example[1])**2
        return (loss/len(batch))**0.5