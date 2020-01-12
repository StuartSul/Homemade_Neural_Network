from hnn import hnn
from tests import test_global, test_classification, test_regression

num_hidden_layer = 10;
node_per_layer = 20;
batch_size = 5;
learning_rate = .1;
total_epochs = 2000;
periods = 20;

#test_classification.generate(0, 1, 1000)
#features, labels = test_global.load_data(test_global.default_filename_classification)
test_regression.generate(0, 1, 1000)
features, labels = test_global.load_data(test_global.dafault_filename_regression)

hnn.init("TEST", len(features[0]), num_hidden_layer,
         node_per_layer, features, labels, 0.8)
hnn.train(batch_size, learning_rate,
           total_epochs, periods)
           