# Code to simgle layer multi neuron Neural Network

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