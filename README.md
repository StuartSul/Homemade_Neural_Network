# Homemade Neural Network

This is a simple neural network made without using any external libraries. One might find this useful in educational circumstances.

To run training and see results, simply execute following command on terminal:

```shell
python3 main.py
```

To tweak some hyper-parameters and use other data, you need to change some codes in main.py.

---
###main.py
Defines necessary hyper parameters and input data, and runs an instance of Trainer.

###network.py
Defines class Network, Layer, and Node, which overall describe neural network structure and functionalities.

###trainer.py
Defines class Trainer, which trains a given network for set amount of time.

###model.py
Saves and loads an instance of network model.


---
##History
Jan 10, 2019: Project started.
Jan 11, 2019: Constructed basic framework.