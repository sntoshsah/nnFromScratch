import numpy as np


# n_inputs = 4
X = [
    [1,2,3,3.4],
    [2,1,4,-3.6],
    [6,-4,3,-2]
]
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print("Output of First layer\n",layer1.output)
layer2.forward(layer1.output)
print("Output of second Layer\n",layer2.output)
