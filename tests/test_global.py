primary_separator = '\n'
secondary_separator = ','

default_filename_classification = 'tests/test_classification.data'
dafault_filename_regression = 'tests/test_regression.data'

def load_data(filename):
    with open(filename) as data_file:
        data_set = data_file.read()
    data_set = data_set.split(primary_separator)
    data_set = [x.split(secondary_separator) for x in data_set]

    features = [[float(x) for x in [*y[:-1]]] for y in data_set]
    labels = [float(x[-1]) for x in data_set]

    return features, labels
    