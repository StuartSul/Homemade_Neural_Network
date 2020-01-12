# Homemade Neural Network

This is a simple neural network made without using any external libraries. One might find this useful in educational circumstances.

To run test network training and see results, simply execute following command on terminal:

```
python3 runtests.py
```

To train and build network for your own use, you may import hnn on your program and use functionalities in hnn.hnn:

``` python3
from hnn import hnn

hnn.init(NETWORK_NAME, INPUT_WIDTH, NUMBER_OF_LAYERS,
         NODES_PER_LAYER, FEATURES, LABELS, TRAIN_TEST_RATIO)

hnn.train(BATCH_SIZE, LEARNING_RATE, TOTAL_EPOCHS, PERIODS)

hnn.predict(NEW_DATA)
```

To be added on pip.

---
### hnn/
Package folder. Defines all network properties, training system, file IO, and other functionalities needed. All APIs are defined in hnn.py.

### tests/
Contains test codes for debugging.

### runtests.py
Runs test codes in tests folder.

---
## History
Jan 10, 2019: Project started.