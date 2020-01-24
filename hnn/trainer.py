from random import random, sample

from hnn.network import Network

class Trainer:
    def __init__(self, network, features, labels, train_ratio, loss_function, regularization=None):
        self.network = network
        self.features = features
        self.labels = labels
        self.train_set, self.test_set = self.divide_data(train_ratio)
        self.loss_function = loss_function

    def train(self, batch_size, learning_rate, epochs):
        for i in range(epochs):
            batch = sample(self.train_set, batch_size)
            gradients = []
            for train_example in batch:
                prediction = self.network.execute(train_example[0])[0]
                label = train_example[1]
                gradients.append(self.backpropagate(prediction, label))
            self.modify_network(gradients, learning_rate)
        train_loss = self.get_loss(self.train_set)
        test_loss = self.get_loss(self.test_set)
        return train_loss, test_loss
    
    def backpropagate(self, prediction, label):
        gradients = []
        derivatives = [[self.loss_function.derivative(prediction, label)]]
        for i in range(len(self.network.layers) - 1, -1, -1):
            layer = self.network.layers[i]
            gradients.insert(0, [])
            next_derivatives = [[]] * layer.input_count
            for j in range(layer.node_count):
                node = layer.nodes[j]
                for k in range(len(derivatives[j])):
                    derivatives[j][k] *= layer.activation.derivatives(layer.output[j])
                derivatives[j] = sum(derivative[j])
                gradients[0].append(list(map(lambda x: x * derivatives[j], layer.input)))
                for k in range(layer.input_count):
                    next_derivatives[k].append(node.weights[k] * derivatives[j])
            derivatives = next_derivatives
        return gradients
            
    def modify_network(self, gradients, learning_rate):
        for example in gradients:
            for layer_index in range(len(example):
                for node_index in range(len(example[layer_index])):
                    for gradient_index in range(len(example[layer_index][node_index]):)
                        self.network.layers[layer_index].nodes[node_index].weights[gradient_index] -=\
                            learning_rate * example[layer_index][node_index][gradient_index] / len(gradients)

    def get_loss(self, data_set):
        predictions = []
        labels = []
        for test_example in data_set:
            predictions.append(self.network.execute(test_example[0])[0])
            labels.append[test_example[1]]
        loss = self.loss_function.calculate(predictions, labels)
        return loss

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