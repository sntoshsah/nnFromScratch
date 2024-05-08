from implementLoss import Layer_Dense, Activation_Relu, Activation_Softmax, Loss_CategoricalCrossEntropy
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("Agg")
import nnfs
from nnfs.datasets import vertical_data
import numpy as np

"""
1. Imports:

    From the implementLoss file, it imports the classes defining the network layers ( Layer_Dense ), activation functions ( Activation_Relu, Activation_Softmax ), and the loss function ( Loss_CategoricalCrossEntropy ).
    Other imports handle plotting (not used here) and data generation.

2. Data Initialization:

    nnfs.init(): Initializes the libraries used (likely for setting seeds or global configurations).
    X, y = vertical_data(samples=100, classes=3): Generates a dataset (X, y) using the vertical_data function, likely creating a dataset with two features (hence "vertical") for 100 samples and three possible classes.

3. Network Architecture:

    dense1 = Layer_Dense(2,3): Creates the first dense layer with 2 input neurons (referencing the two features) and 3 hidden neurons.
    activation1 = Activation_Relu(): Creates a ReLU activation layer.
    dense2 = Layer_Dense(3,3): Creates the second dense layer with 3 input neurons (number of neurons in the previous layer) and 3 output neurons (one for each class).
    activation2 = Activation_Softmax(): Creates a Softmax activation layer for the output layer, converting the outputs into class probabilities.
    loss_function = Loss_CategoricalCrossEntropy(): Creates an instance of the Categorical Cross-Entropy loss function.

4. Training Loop (Stochastic Gradient Descent):

    lowest_loss = 999999: Initializes a variable to store the lowest loss encountered during training.
    Initializes variables to store the best weights and biases found so far for each layer.
    The loop iterates for 100,000 training steps:
        dense1.weights += 0.05 * np.random.randn(2,3): Updates the weights in the first dense layer with a small random value (learning rate 0.05) multiplied by random noise. This stochastic (random) update helps the network escape local minima during optimization.
        Similar updates are performed for biases in both layers (dense1.biases, dense2.biases) and weights in the second layer (dense2.weights).
        Forward pass through the network:
            dense1.forward(X): Propagates the input data through the first dense layer.
            activation1.forward(dense1.output): Applies the ReLU activation to the first layer's output.
            dense2.forward(activation1.output): Propagates the output from the activation layer through the second dense layer.
            activation2.forward(dense2.output): Applies the Softmax activation to the output layer, generating class probabilities.
        loss = loss_function.calculate(activation2.output, y): Calculates the loss based on the predicted probabilities (activation2.output) and the true target labels (y).
        predictions = np.argmax(activation2.output, axis=1): Converts the softmax outputs into predicted class labels by finding the index of the highest probability for each sample.
        accuracy = np.mean(predictions==y): Calculates the accuracy of the predictions by comparing them to the true labels (y).
        if loss < lowest_loss : Checks if the current loss is lower than the previously encountered lowest loss.
            If yes, it updates the best_*_weights and best_*_biases variables to store the current weights and biases (potentially leading to better performance).
            It also updates lowest_loss to reflect the new lowest value encountered.
"""

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

# plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
# plt.show()

dense1 = Layer_Dense(2,3)
activation1 = Activation_Relu()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(100000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    if loss < lowest_loss:
        print("New set of weights found, iteration: ", iteration, "Loss: ",loss, "acc: ", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

