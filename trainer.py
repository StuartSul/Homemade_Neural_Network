from random import random, sample
from math import log

import network

class Trainer:
    loss_modifier = 30

    def __init__(self, network, features, labels, batch_size=5, train_ratio=0.8):
        self.network = network
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.train_set, self.test_set = self.divide_data()

    def train(self, step, reps):
        for i in range(0, reps):
            batch = sample(self.train_set, self.batch_size)

            original_loss = self.run_batch(batch)
            self.network.modify(-step, shuffle=True)
            down_loss = self.run_batch(batch)
            self.network.modify(2 * step)
            up_loss = self.run_batch(batch)
            
            if down_loss >= original_loss and up_loss >= original_loss:
                self.network.modify(-step)
            elif down_loss < up_loss:
                self.network.modify(-2 * step)
                
        test_loss = self.run_batch(self.test_set)
        return test_loss
    
    def divide_data(self):
        size = len(self.features)
        train_size = int(size * self.train_ratio)

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
            loss += Trainer.get_loss(result, example[1])
        return loss
    
    @staticmethod
    def get_loss(result, label):
        if label == 'H':
            if result < 0:
                return -result * Trainer.loss_modifier
            else:
                return -log(result + 1)
        else:
            if result >= 0:
                return result * Trainer.loss_modifier
            else:
                return -log(-result + 1)