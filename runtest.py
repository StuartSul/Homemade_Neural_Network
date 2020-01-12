from hnn import hnn

hyper_param = {
    "feature_size": 2,
    "hidden_layer": 20,
    "label_size": 1,
    "batch_size": 5,
    "node_per_layer": 20,
    "step": .1,
    "rep": 500,
    "period": 50
}

file_name = 'tests/data.csv'
with open(file_name) as data_file:
    data_set = data_file.read()
data_set = data_set.split('\n')
data_set = [x.split(',') for x in data_set]
data_set = data_set[0:-1]

features = [[float(x[0]), float(x[1])] for x in data_set]
labels = [float(x[2]) for x in data_set]

hnn.init("TEST", len(features[0]), hyper_param["hidden_layer"],
         hyper_param["node_per_layer"], features, labels, 
         0.8)
hnn.train(hyper_param["batch_size"], hyper_param["step"],
           hyper_param["rep"], hyper_param["period"])