"""
log is solving the value of x from equation,  where e**x = b

"""
"""
1. Logarithm and Exponentiation:

    b = 10: Assigns the value 10 to variable b.
    print(np.log(b)): Calculates the natural logarithm (base-e) of b using np.log(b). The natural logarithm is the inverse of exponentiation with base e (approximately 2.718). So, finding the log of b essentially asks "what power of e raised to it equals b?". In this case, np.log(10) is approximately 2.303.
    print(math.e**(np.log(b))): Calculates e raised to the power of np.log(b). Since the logarithm is the inverse of exponentiation, this effectively reverses the previous step and should ideally return the original value b (due to rounding errors, it might not be exactly 10).

2. Categorical Cross-Entropy Loss (Simplified):

    This section demonstrates a simplified calculation of the Categorical Cross-Entropy Loss function, commonly used in multi-class classification problems.
    softmax_output = [0.7, 0.1, 0.2]: Defines a list softmax_output containing three values, likely representing the probabilities for three classes predicted by a softmax activation function (they should sum to 1).
    target_output = [1, 0, 0]: Defines a list target_output representing the desired output (one-hot encoded), where only the element corresponding to the correct class is 1 (here, class 0), and the others are 0.
    The following lines calculate the loss:
        The original calculation uses a loop-like approach with math.log for each element-wise multiplication. It iterates through the elements of softmax_output and target_output, multiplying the corresponding log of the softmax probability with the target value. Since the target value for the correct class (index 0) is 1, its corresponding term dominates the calculation.
        A simplified version calculates the loss only for the correct class (index 0), focusing on the negative log of the predicted probability for that class (-math.log(softmax_output[0])). This is because the target value for the correct class is 1, and the other target values are 0, making their contribution to the loss zero.
"""

import numpy as np
import math

b = 10

print(np.log(b)) # log of 10

print(math.e**(np.log(b))) 

# loss with categorical cross entropy

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2] 
         )

print(loss) # -math.log(softmax_output[0] + 0 + 0)
loss = -math.log(softmax_output[0])
print(loss)
print(-math.log(0.7))
print(-math.log(0.5))
print(-math.log(0.2))

