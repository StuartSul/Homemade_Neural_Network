from inspect import signature
from random import random

from .test_global import *

# y = x
def sample_relation(x):
    return x

def generate(min, max, size, relation=sample_relation, 
                filename=dafault_filename_regression, save=True):
    input_width = len(signature(relation).parameters)
 
    data = []
    for i in range(size):
        data.append([])
        for j in range(input_width):
            data[i].append((max - min) * random() + min)
        data[i].append(relation(*data[i]))

    data_str = ''
    for example in data:
        for value in example:
            data_str += str(value) + secondary_separator
        data_str = data_str[:len(data_str) - len(secondary_separator)]
        data_str += primary_separator
    data_str = data_str[:len(data_str) - len(primary_separator)]

    if save:
        with open(filename, "w") as data_file:
            data_file.write(data_str)
    
    return data_str