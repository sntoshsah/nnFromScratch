import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nnfs.datasets import spiral_data
matplotlib.use('Agg')

np.random.seed(0)

"""
1. Data Import:

    It imports the necessary libraries:
        numpy as np: For numerical computations.
        matplotlib.pyplot as plt: For plotting visualizations (used later).
        matplotlib: Additional matplotlib functionality.
        nnfs.datasets import spiral_data: Imports the spiral_data function to generate a two-class spiral dataset commonly used for binary classification tasks.
    np.random.seed(0): Sets a random seed for reproducibility (ensuring the same random numbers are generated each time the code runs).

2. Commented Out Data Creation (Optional):

    The commented-out section (# def create_data(points, classes):...) defines a function create_data to generate custom data. This functionality is replaced by using the spiral_data function for this specific example.

3. Data Generation:

    X, y = spiral_data(100, 3): Generates a spiral dataset with 100 data points and 3 dimensions (2 features and likely a bias term). This dataset is specifically designed for visualizing and evaluating binary classification problems. The X variable holds the data points, and y contains the corresponding class labels (0 or 1 for the two classes in the spiral).

4. Class Definitions:

    Maintained Classes:
        Layer_Dense: This class definition remains the same, representing a dense layer in the neural network.
    New Class:
        Activation_Relu: This new class defines the ReLU (Rectified Linear Unit) activation function.
            forward(self, inputs): This method takes the input values and applies the ReLU function element-wise. ReLU sets any negative value to zero, effectively introducing a threshold into the activation.

5. Network Creation and Forward Pass:

    layer1 = Layer_Dense(2, 5): Creates a dense layer named layer1 with 2 input neurons (referencing the two features in the spiral data) and 5 hidden neurons.
    activation1 = Activation_Relu(): Creates an instance of the Activation_Relu class, naming it activation1.
    layer1.forward(X): Performs the forward pass through the first dense layer (layer1), calculating the weighted sum of inputs for each hidden neuron.
    print('Before Activation'): Prints the output before the ReLU activation (the weighted sum of inputs and biases for each neuron in layer1).
    activation1.forward(layer1.output): Applies the ReLU activation function to the output of the first layer (layer1.output).
    print("After Activation"): Prints the output after the ReLU activation, showing how the ReLU function sets negative values to zero.
"""


# def create_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points) # radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X,y

# print('Here')
# X,y = create_data(100,3)

# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.plot()
# plt.show()

X,y = spiral_data(100,3)


# # n_inputs = 4
# X = [
#     [1,2,3,3.4],
#     [2,1,4,-3.6],
#     [6,-4,3,-2]
# ]
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2,5)
activation1 = Activation_Relu()

layer1.forward(X)
print('Before Activation')
print(layer1.output)
activation1.forward(layer1.output)
print("After Activation")
print(activation1.output)