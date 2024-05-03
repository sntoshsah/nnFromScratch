inputs = [1.2, 5.1, 2.1] # x - values
weights = [3.1, 2.1, 8.7]

bias = 3
# y = a0x0 + a1x1 + a2x2 + b
outputs = inputs[0]*weights[0]+ inputs[1]* weights[1]+ inputs[2]*weights[2] +bias

print(outputs)