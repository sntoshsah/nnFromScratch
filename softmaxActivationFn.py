import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nnfs.datasets import spiral_data
matplotlib.use('Agg')

np.random.seed(0)


"""
1. Library Imports:

    Similar to previous examples,

    it imports necessary libraries for numerical computations (numpy - np), plotting (matplotlib.pyplot - plt), and functionalities related to neural networks (nnfs.datasets).

2. Data Generation:

    np.random.seed(0): Sets a random seed for reproducibility.
    X, y = spiral_data(100, 3): Generates a spiral dataset with 100 data points and 3 dimensions (2 features and likely a bias term). This dataset is commonly used for visualizing and evaluating binary classification problems. X holds the data points, and y contains the corresponding class labels (0 or 1 for the two classes in the spiral).

3. Class Definitions:

    The code defines three classes for the network layers:
        Layer_Dense: This class represents a dense layer in the network, taking the number of input neurons and the number of neurons in the layer as arguments. It initializes random weights and biases using the __init__ method and performs a linear transformation of the input during the forward pass.
        Activation_Relu: This class implements the ReLU (Rectified Linear Unit) activation function. The forward method applies the ReLU function element-wise, setting any negative value to zero.
        Activation_Softmax: This class defines the Softmax activation function. The forward method calculates the softmax function, which converts the output from the final layer into class probabilities that sum to 1. This is essential for multi-class classification.

4. Network Creation and Forward Pass:

    dense1 = Layer_Dense(2,5): Creates the first dense layer with 2 input neurons (referencing the two features in the data) and 5 hidden neurons.
    activation1 = Activation_Relu(): Creates an instance of the Activation_Relu class for the activation function after the first layer.
    dense2 = Layer_Dense(5,3): Creates the second dense layer with 5 input neurons (number of neurons in the previous layer) and 3 output neurons (one for each class).
    activation2 = Activation_Softmax(): Creates an instance of the Activation_Softmax class for the activation function in the output layer.

Partial Forward Pass:

    dense1.forward(X): Performs the forward pass through the first dense layer, calculating the weighted sum of inputs for each hidden neuron.
    Prints show the state before and after the ReLU activation:
        print('Before relu Activation'): Prints the output of the first layer before applying the ReLU activation (the weighted sum of inputs and biases for each neuron).
        activation1.forward(dense1.output): Applies the ReLU activation to the output of the first layer.
        print("After relu Activation"): Prints the output after the ReLU activation, demonstrating how negative values become zero.

Limited Forward Pass (for demonstration):

    dense2.forward(activation1.output[:5]): Performs the forward pass through the second dense layer, but only for the first 5 samples in the batch (activation1.output[:5]). This is likely for demonstration purposes, as a full forward pass would use all data points in X.
    Prints show the state before and after the Softmax activation:
        print('Before softmax Activation'): Prints the output of the second layer before applying the Softmax activation.
        activation2.forward(dense2.output): Applies the Softmax activation to the output of the second layer (limited to the first 5 samples).
        print("After softmax Activation"): Prints the output after the Softmax activation, showing the class probabilities for the first 5 samples.

In essence, this code showcases a two-layer neural network architecture with ReLU and Softmax activations. It demonstrates a partial forward pass through the network on a spiral dataset, highlighting the effect of each activation function on the output.
"""


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