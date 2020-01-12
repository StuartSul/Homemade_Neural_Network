from inspect import signature
from random import random

from .test_global import *

classification_weight = 10

# y = x
def sample_divider(x, y):
    return y - x

def generate(min, max, size, divider=sample_divider, filename=default_filename_classification):
    input_width = len(signature(divider).parameters)
    while True:
        data = []
        count = 0

        for i in range(size):
            data.append([])

            for j in range(input_width):
                data[i].append((max - min) * random() + min)

            if divider(*data[i]) >= 0:
                data[i].append(classification_weight)
                count += 1
            else:
                data[i].append(-classification_weight)

        if count >= size * 0.45 and count <= size * 0.55:
            break

    data_str = ''
    for example in data:
        for value in example:
            data_str += str(value) + secondary_separator
        data_str = data_str[:len(data_str) - len(secondary_separator)]
        data_str += primary_separator
    data_str = data_str[:len(data_str) - len(primary_separator)]

    with open(filename, "w") as data_file:
        data_file.write(data_str)
    
    return data_str