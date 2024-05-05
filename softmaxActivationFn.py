import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nnfs.datasets import spiral_data
matplotlib.use('Agg')

np.random.seed(0)



X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


dense1 = Layer_Dense(2,5)
activation1 = Activation_Relu()

dense2 = Layer_Dense(5,3)
activation2 = Activation_Softmax()

dense1.forward(X)
print('Before  relu Activation')
print(dense1.output)
activation1.forward(dense1.output)
print("After relu Activation")
print(activation1.output)

dense2.forward(activation1.output[:5])
print('Before  softmax Activation')
print(dense2.output)
activation2.forward(dense2.output)
print("After softmax Activation")
print(activation2.output[:5])