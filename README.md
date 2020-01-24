# Homemade Neural Network

This is a simple neural network made without using any external libraries. One might find this useful in educational circumstances. Python 3.8 was used.

To run test network training and see results, simply execute the following command on terminal:

```
python3 runtests.py
```

To generate your own test data with user-defined data relation, use the modules provided in tests folder:

``` python3
from tests import test_global, test_classification, test_regression

# Generates classification test data
test_classification.generate(DATA_SIZE, divider=YOUR_FUNCTION)
features, labels = test_global.load_data(test_global.default_filename_classification)

# Generates regression test data
test_regression.generate(DATA_SIZE, relation=YOUR_FUNCTION)
features, labels = test_global.load_data(test_global.dafault_filename_regression)

```

To train and build network for your own use, you may import hnn on your program and use functionalities in hnn.hnn:

``` python3
from hnn.hnn import hnn

my_hnn = hnn(NETWORK_NAME, INPUT_COUNT, OUTPUT_COUNT, STRUCTURE, ACTIVATION
                FEATURES, LABELS, TRAIN_TEST_RATIO, LOSS_FUNCTION)

my_hnn.train(BATCH_SIZE, LEARNING_RATE, TOTAL_EPOCHS, PERIODS)

my_hnn.predict(NEW_DATA)
```

ACTIVATION and LOSS_FUNCTION are defined under hnn/util.py, and structure should be a list of integers: the length of list corresponding to total number of layers and each integer value corresponding to number of nodes in each layer. For instance:

``` python3
from hnn.hnn import hnn
import hnn.util

my_hnn = hnn('TEST', 4, 1, [4, 4, 4], hnn.util.Sigmoid()
                FEATURES, LABELS, 0.8, hnn.util.RMSE())

my_hnn.train(5, 0.3, 10000, 20)

my_hnn.predict(NEW_DATA)
```

Once trained, you can save and load your network via file IO:

``` python3
my_hnn.save(FILE_NAME)

new_hnn = hnn.load(FILE_NAME)

new_hnn.predict(NEW_DATA)
```

You can try sample network models provided in models folder:

``` python3
new_hnn = hnn.load('models/x=y_classification.hnn')
new_hnn.predict(NEW_DATA)

new_hnn = hnn.load('models/x=y_regression.hnn')
new_hnn.predict(NEW_DATA)
```

---
### hnn/
Package folder. Defines all network properties, training system, file IO, and other functionalities needed. All APIs are defined in hnn.py.

### tests/
Contains test codes for debugging.

### runtests.py
Runs test codes in tests folder.

### models/
Contains sample trained models.

---
## History
Jan 10, 2019: Project started.