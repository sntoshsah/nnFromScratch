"""Explanation:

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
"""


inputs = [1.2, 5.1, 2.1] # x - values
weights = [3.1, 2.1, 8.7]

bias = 3
# y = a0x0 + a1x1 + a2x2 + b
outputs = inputs[0]*weights[0]+ inputs[1]* weights[1]+ inputs[2]*weights[2] +bias

print(outputs)