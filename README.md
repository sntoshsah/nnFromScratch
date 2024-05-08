# This repository is the demonstrated code of Neural Network From Scratch.
The detailed Structure of the repository is as follows :
1. test
Explanation:

    Data Initialization:
        inputs: This line creates a list named inputs containing three numbers: 1.2, 5.1, and 2.1. These numbers represent the x-values (often called features or independent variables) that will be used in a calculation.
        weights: Similarly, weights is a list containing three numbers: 3.1, 2.1, and 8.7. These weights correspond to the coefficients of the x-values in a linear equation. They influence the importance of each x-value in the calculation.
        bias: The variable bias is assigned the value 3. The bias term is a constant value that is added to the final result of the calculation, regardless of the input values. It can be used to adjust the overall output of the equation.

    Calculation:
        The line outputs = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias performs the core calculation. It multiplies each element in the inputs list with its corresponding element in the weights list, adds those products together, and then adds the bias value.
            inputs[0] refers to the first element (1.2) in the inputs list.
            weights[0] refers to the first element (3.1) in the weights list.
            This multiplication and addition process continues for all three elements in the lists.

    Output:
        Finally, the result of the calculation (outputs) is printed using the print function. The output will depend on the specific values in inputs, weights, and bias.

In essence, this code snippet implements a linear equation:

y = a0 * x0 + a1 * x1 + a2 * x2 + b

    Where:
        y represents the output value (stored in the outputs variable).
        a0, a1, and a2 are the weights (corresponding to the elements in the weights list).
        x0, x1, and x2 are the input values (elements in the inputs list).
        b is the bias term (bias variable).

2. codeLayer
Data Initialization:

    It creates an input list x containing four values: [1, 2, 3, 2.5]. These represent the inputs to the neural network.
    Three weight lists are defined: w1, w2, and w3. Each list corresponds to the weights for one neuron, and their lengths match the number of elements in the input (x). These weights determine the influence of each input on the neuron's output.
    Three bias values (bias1, bias2, bias3) are set for the three neurons. The bias acts as an additional constant term in the neuron's activation.

Calculation:

    The core calculation happens in the output list creation. It iterates through the three neurons:
        For each neuron (represented by w1, w2, or w3), it calculates the weighted sum of the inputs. This is done by multiplying each element in the input list (x) with its corresponding weight in the current weight list (w1, w2, or w3) and summing those products.
        The bias value for the corresponding neuron is then added to this weighted sum.

Output:

    The output list stores the results of these calculations for each neuron. This list will contain three numbers, representing the outputs of the three neurons in the network.

3. layerbyDotProduct
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
    print(np.dot(weights, input) + biases): This line demonstrates an alternative approach using NumPy's efficient dot product function (np.dot). It calculates the matrix multiplication between weights and input, and then adds the biases element-wise. The result is printed.
4. batchesAndLayers
5. layersClass
6. layerswithActivationFn(relu)
7. softmaxActivationFn
8. lossWithCrossEntropy
9. implementLoss
10. Optimization
11. Optimization with tangent