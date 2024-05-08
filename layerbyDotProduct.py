import numpy as np

"""
Library Import:

   import numpy as np: This line imports the NumPy library, which provides powerful functions for numerical computations. It's assigned the alias np for convenience.

Data Initialization:

   input = [1, 3, 5, 6]: An input list input is created containing four numbers: 1, 3, 5, and 6. These represent the inputs to the neural network.
  weights: A list named weights is defined. It contains three sub-lists, each representing the weights for a single neuron in the layer. The length of each sub-list (4 in this case) matches the number of elements in the input (input). These weights determine the influence of each input on the neuron's output.
  biases: A list biases is created containing three bias values, one for each neuron in the layer. The bias term acts as an additional constant added to the neuron's activation.

 Custom Dot Product Function:

   def customDotProduct(input, weights, biases):: This defines a function named customDotProduct that takes three arguments: input (the input list), weights (the list of weight lists), and biases (the list of biases).
  Inside the function:
      An empty list layer_outputs is initialized to store the outputs of each neuron.
      It iterates through the elements in weights and biases using zip(), which pairs corresponding elements from two lists. So, in each iteration, neuron_weight will hold a weight list for a neuron, and neuron_bias will hold the corresponding bias value for that neuron.
      For each neuron:
          A variable neuron_output is initialized to 0 to accumulate the weighted sum of inputs for that neuron.
          It iterates through the elements in input and neuron_weight using another zip(). Here, n_input will hold an input value and weight will hold the corresponding weight from the current neuron_weight list.
          Within this inner loop, it performs neuron_output += n_input * weight, essentially calculating the dot product between the input vector and the weight vector for the current neuron.
          The bias value for the current neuron is then added to neuron_output.
          The final neuron_output (the output of the current neuron) is appended to the layer_outputs list.
      The function returns the layer_outputs list containing the outputs of all neurons in the layer.

 Calculation Using Custom Function and NumPy:

    print(customDotProduct(input, weights, biases)): This line calls the customDotProduct function with the defined input, weights, and biases to calculate the outputs using the custom implementation. The result (a list of neuron outputs) is printed.
    print(np.dot(weights, input) + biases): This line demonstrates an alternative approach using NumPy's efficient dot product function (np.dot). It calculates the matrix multiplication between weights and input, and then adds the biases element-wise. The result is printed."""

input = [1,3,5,6]

weights = [[0.2, 0.5, -0.6, 5],
           [0.5, 0.8, 0.9, 1],
           [-1.5, 2.5, 0.7, 0.3]]

biases = [2,3,0.6]
def customDotProduct(input, weights, biases):
    layer_outputs = []
    for neuron_weight, neuron_bias in zip(weights, biases):
        neuron_output = 0 # Output of given neuron
        for n_input, weight in zip(input, neuron_weight):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)
    return layer_outputs

print(customDotProduct(input, weights, biases)) 

print(np.dot(weights,input) + biases) # Using Dot product