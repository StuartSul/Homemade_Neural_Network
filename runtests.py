from hnn.hnn import hnn
from hnn import util
from tests import test_global, test_classification, test_regression

features, labels = test_global.load_data(
            test_classification.generate(2000, save=False),
            isfile=False)

network_id = 'Test_Network'
input_count = len(features[0])
output_count = 1
structure = [4, 4]
activation = util.Sigmoid()

train_ratio = 0.8
loss_function = util.LogLoss()

batch_size = 5
learning_rate = 0.03
total_epochs = 50000
periods = 20

my_hnn = hnn(network_id, input_count, output_count, structure, activation, 
                        features, labels, train_ratio, loss_function)

my_hnn.train(batch_size, learning_rate, total_epochs, periods)