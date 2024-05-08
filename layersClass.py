import numpy as np

"""
1. Library Import:

    import numpy as np: Imports the NumPy library (np), which provides powerful functions for numerical computations like matrix operations, essential for neural networks.

2. Data Initialization:

    X = [[1,2,3,3.4], [2,1,4,-3.6], [6,-4,3,-2]]: Creates a list named X containing three sub-lists. Each sub-list represents a single input vector with four elements. These input vectors will be fed into the neural network.

3. Class Definition:

    class Layer_Dense**: This class defines a dense layer, a fundamental building block in neural networks.
        __init__(self, n_inputs, n_neurons): The constructor of the class takes two arguments:
            n_inputs: The number of input neurons in the layer (usually determined by the number of features in the input data).
            n_neurons: The number of neurons in the layer.
        weights = 0.10 * np.random.randn(n_inputs, n_neurons): Initializes the weights attribute of the layer. This is a random weight matrix with dimensions (n_inputs x n_neurons). The weights determine the influence of each input on the output of each neuron in the layer. The np.random.randn function generates a random array with a standard normal distribution (mean=0, standard deviation=1), and it's scaled by 0.10 here.
        biases = np.zeros((1, n_neurons)): Initializes the biases attribute of the layer. This is a bias vector with one value for each neuron (shape: 1 x n_neurons). The bias term acts as an additional constant added to the activation of each neuron. Here, it's initialized with zeros.
        forward(self, inputs): This method performs the forward pass through the layer:
            self.output = np.dot(inputs, self.weights) + self.biases: Calculates the weighted sum of inputs for each neuron. It uses the np.dot function for matrix multiplication between the input data (inputs) and the weight matrix (self.weights). The bias vector (self.biases) is then added element-wise to this result.

4. Network Creation:

    layer1 = Layer_Dense(4, 5): Creates an instance of the Layer_Dense class, naming it layer1. It specifies 4 input neurons (referencing the four elements in each input vector) and 5 neurons in this first hidden layer.
    layer2 = Layer_Dense(5, 2): Creates another instance of the Layer_Dense class, naming it layer2. It specifies 5 input neurons (referencing the number of neurons in the previous layer) and 2 neurons in the output layer. This two-neuron output layer likely signifies a binary classification task (predicting one of two classes).

5. Forward Pass:

    layer1.forward(X): Performs the forward pass through the first layer (layer1). It calls the forward method of layer1, passing the input data (X) as the argument. This calculates the weighted sum of inputs and adds the bias for each neuron in layer1, storing the results in the layer1.output attribute.
    print("Output of First layer\n", layer1.output): Prints the output of the first layer (layer1.output), which represents the activations of the hidden neurons after the first layer's processing.
    layer2.forward(layer1.output): Performs the forward pass through the second layer (layer2). It uses the output of the first layer (layer1.output) as the input for layer2. This effectively propagates the information through the network.
    print("Output of second Layer\n", layer2.output): Prints the output of the second layer (layer2.output), which represents the final activations of the output layer neurons.
"""
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
