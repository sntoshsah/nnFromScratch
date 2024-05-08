import numpy as np

# softmax_outputs = np.array([[0.0, 0.1, 0.3],  # -log(0) gives infinity
#                             [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08]
#                             ])

# class_targets = [0,1,1]

# print(-np.log(softmax_outputs[[0,1,2],[class_targets]]))
# print(np.mean(-np.log(softmax_outputs[[0,1,2], class_targets])))


"""
1. Imports:

    numpy as np: Imports the NumPy library for numerical computations.
    matplotlib.pyplot as plt: Imports the plotting library for visualization (though not used in this specific execution).
    nnfs.datasets import spiral_data: Imports the spiral_data function from the nnfs library (likely a custom library for neural network research) to generate the spiral dataset for classification.

2. Class Definitions:

    Layer_Dense: This class defines a dense layer in a neural network.
        __init__: Initializes the layer with a random weight matrix and a bias vector. The weights determine the influence of each input on the neuron's output, and the bias term is an additional constant added to the activation.
        forward: Takes the input data and performs the matrix multiplication between the weights and the input, followed by adding the bias. This calculates the weighted sum of inputs for each neuron in the layer.
    Activation_Relu: This class implements the ReLU (Rectified Linear Unit) activation function.
        forward: Applies the ReLU function element-wise to the input. ReLU sets any negative value to zero, essentially creating a threshold activation.
    Activation_Softmax: This class implements the Softmax activation function.
        forward: Takes the input, performs calculations to normalize the values and ensure they sum to 1, effectively converting them into probabilities for a multi-class classification problem.
    Loss: This is an abstract class representing a loss function.
        calculate: This method, to be implemented by subclasses, calculates the loss based on the predicted output and the true target values.
    Loss_CategoricalCrossEntropy: This class inherits from Loss and implements the Categorical Cross-Entropy loss function commonly used in multi-class classification problems.
        forward: Calculates the cross-entropy loss for each sample and returns the mean loss across all samples.

3. Main Function (main):

    X, y = spiral_data(100, 3): Generates a spiral dataset with 100 data points and 3 dimensions (2 features and likely a bias term). This dataset is commonly used for visualizing binary classification problems.
    Network Creation:
        dense1 = Layer_Dense(2, 5): Creates a dense layer with 2 input neurons (likely corresponding to the two features in the data) and 5 hidden neurons.
        activation1 = Activation_Relu(): Creates a ReLU activation layer.
        dense2 = Layer_Dense(5, 3): Creates another dense layer with 5 input neurons (number of neurons in the previous layer) and 3 output neurons (one for each class in the binary classification).
        activation2 = Activation_Softmax(): Creates a Softmax activation layer for the output layer.
    Forward Pass:
        dense1.forward(X): Performs the forward pass through the first dense layer, calculating the weighted sum of inputs for each hidden neuron.
        Prints the output before the ReLU activation.
        activation1.forward(dense1.output): Applies the ReLU activation to the hidden layer outputs.
        Prints the output after the ReLU activation.
        dense2.forward(activation1.output): Performs the forward pass through the second dense layer.
        Prints the output before the Softmax activation.
        activation2.forward(dense2.output): Applies the Softmax activation to the output layer, converting the values into class probabilities.
        Prints the first 5 elements of the Softmax output (likely representing probabilities for the first 5 data points).
    Loss Calculation:
        loss_function = Loss_CategoricalCrossEntropy(): Creates an instance of the Categorical Cross-Entropy loss function.
        loss = loss_function.calculate(activation2.output, y): Calculates the loss based on the Softmax output probabilities and the true target labels (y).
        Prints the calculated loss value.
"""

import matplotlib.pyplot as plt
import matplotlib
from nnfs.datasets import spiral_data
matplotlib.use('Agg')

np.random.seed(0)




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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
    
        return negative_log_likelihoods
    


def main():
    X, y = spiral_data(100, 3)  
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

    dense2.forward(activation1.output)
    print('Before  softmax Activation')
    print(dense2.output)
    activation2.forward(dense2.output)
    print("After softmax Activation")
    print(activation2.output[:5])


    loss_function = Loss_CategoricalCrossEntropy()
    loss = loss_function.calculate(activation2.output, y)

    print("Loss:", loss)

if __name__ == "__main__":
    main()