"""
log is solving the value of x from equation,  where e**x = b

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

