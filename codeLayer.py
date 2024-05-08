"""Data Initialization:

    It creates an input list x containing four values: [1, 2, 3, 2.5]. These represent the inputs to the neural network.
    Three weight lists are defined: w1, w2, and w3. Each list corresponds to the weights for one neuron, and their lengths match the number of elements in the input (x). These weights determine the influence of each input on the neuron's output.
    Three bias values (bias1, bias2, bias3) are set for the three neurons. The bias acts as an additional constant term in the neuron's activation.

Calculation:

    The core calculation happens in the output list creation. It iterates through the three neurons:
        For each neuron (represented by w1, w2, or w3), it calculates the weighted sum of the inputs. This is done by multiplying each element in the input list (x) with its corresponding weight in the current weight list (w1, w2, or w3) and summing those products.
        The bias value for the corresponding neuron is then added to this weighted sum.

Output:

    The output list stores the results of these calculations for each neuron. This list will contain three numbers, representing the outputs of the three neurons in the network."""
# Code to single layer multi neuron Neural Network

x = [1,2,3,2.5]
w1 = [0.3, 0.4, -0.2, 2] # 1st neuron's Weights
w2 = [0.5, -0.91, 0.26, -0.5] # 2nd Neuron's weights
w3 = [-0.25, 0.35, 0.17, 0.87] # 3rd Neuron's weights
            
bias1 = 3 # for neuron1
bias2 = 2 # for neuron2
bias3 = 0.45 # for neuron3

output = [
    x[0] * w1[0] + x[1]*w1[1] + x[2]*w1[2] + x[3]*w1[3] + bias1,
    x[0] * w2[0] + x[1]*w2[1] + x[2]*w2[2] + x[3]*w2[3] + bias2,
    x[0] * w3[0] + x[1]*w3[1] + x[2]*w3[2] + x[3]*w3[3] + bias3,
]

print(output)