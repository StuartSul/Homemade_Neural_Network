from random import random

# y = f(x)
def ground_truth(x):
    return (x)

data_size = 1000

while True:
    data = [[10 * random(), 10 * random()] for x in range(data_size)]
    count = 0

    for item in data:
        if ground_truth(item[0]) <= item[1]:
            item.append(10)
            count += 1
        else:
            item.append(-10)

    if count >= data_size * 0.45 and count <= data_size * 0.55:
        break

data_str = ''

for item in data:
    data_str += str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + '\n'
 
data_file = open("data.csv", "w")
data_file.write(data_str)
data_file.close()