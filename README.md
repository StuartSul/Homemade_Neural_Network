# Homemade Neural Network

This is a simple neural network made without using any external libraries. One might find this useful in educational circumstances.

To run test network training and see results, simply execute the following command on terminal:

```
python3 runtests.py
```

To generate your own test data with user-defined data relation, use the modules provided in tests folder:

``` python3
from tests import test_global, test_classification, test_regression

# Generates classification test data
test_classification.generate(MIN_FEATURE_VALUE, MAX_FEATURE_VALUE, DATA_SIZE, divider=YOUR_FUNCTION)
features, labels = test_global.load_data(test_global.default_filename_classification)

# Generates regression test data
test_regression.generate(MIN_FEATURE_VALUE, MAX_FEATURE_VALUE, DATA_SIZE, relation=YOUR_FUNCTION)
features, labels = test_global.load_data(test_global.dafault_filename_regression)

```

To train and build network for your own use, you may import hnn on your program and use functionalities in hnn.hnn:

``` python3
from hnn.hnn import hnn

my_hnn = hnn(NETWORK_NAME, INPUT_WIDTH, NUMBER_OF_LAYERS,
                NODES_PER_LAYER, FEATURES, LABELS, TRAIN_TEST_RATIO)

my_hnn.train(BATCH_SIZE, LEARNING_RATE, TOTAL_EPOCHS, PERIODS)

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
new_hnn = hnn.load('x=y_classification.hnn')
new_hnn.predict(NEW_DATA)

new_hnn = hnn.load('x=y_regression.hnn')
new_hnn.predict(NEW_DATA)
```

This will be uploaded on pip sooner or later.

---
### hnn/
Package folder. Defines all network properties, training system, file IO, and other functionalities needed. All APIs are defined in hnn.py.

### tests/
Contains test codes for debugging.

### runtests.py
Runs test codes in tests folder.

### models/
Contains trained sample networks.

---
## History
Jan 10, 2019: Project started.