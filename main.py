import network
import trainer

hyper_param = {
    "feature_size": 2,
    "hidden_layer": 5,
    "label_size": 1,
    "batch_size": 10,
    "node_per_layer": 10,
    "step": 0.005,
    "rep": 5000
}

file_name = 'data.csv'
with open(file_name) as data_file:
    data_set = data_file.read()
data_set = data_set.split('\n')
data_set = [x.split(',') for x in data_set]
data_set = data_set[0:-1]

features = [[float(x[0]), float(x[1])] for x in data_set]
labels = [x[2] for x in data_set]

my_network = network.Network("TEST", 
                        hyper_param["hidden_layer"], 
                        hyper_param["node_per_layer"],
                        len(features[0]))
my_trainer = trainer.Trainer(my_network, features, labels, batch_size=hyper_param["batch_size"])

my_trainer.train(hyper_param["step"], hyper_param["rep"])