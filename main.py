import network
import trainer

from copy import deepcopy

hyper_param = {
    "feature_size": 2,
    "hidden_layer": 10,
    "label_size": 1,
    "batch_size": 50,
    "node_per_layer": 10,
    "step": 0.1,
    "rep": 5000,
    "period": 1000
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

rep_per_period = hyper_param["rep"] // hyper_param["period"]

max_correct = -1
best_network = None

for i in range(hyper_param["period"]):
    print("Period " + str(i+1))
    print("  loss: " + str(my_trainer.train(hyper_param["step"], rep_per_period)))
    correct = 0
    for i in range(len(features)):
        result = my_network.execute(features[i])
        if result > 0 and labels[i] == 'H':
            correct += 1
        elif result < 0 and labels[i] == 'L':
            correct += 1
    print("  result: " + str(correct) + " correct out of " + str(len(features)))
    if (correct > max_correct):
        print("  new best network. Save")
        best_network = deepcopy(my_network)
        max_correct = correct

def p(a,b):
    return my_network.execute([a,b])