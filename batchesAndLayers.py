import numpy as np


inputs = [
    [1,2,3,0.5],
    [3,2,1,-0.5],
    [3,1,2,3],
]

weights = [
    [0.2,0.8,-0.5,0.8],
    [0.4,0.3,0.2,0.1],
    [7,-3,0.3,0.5]
]

biases = [2,3,0.5]

# output = np.dot(inputs, weights) + biases  # dimensional error
output = np.dot(inputs, np.array(weights).T) + biases
print(output)