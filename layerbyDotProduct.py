import numpy as np

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