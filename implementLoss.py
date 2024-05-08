import numpy as np

# softmax_outputs = np.array([[0.0, 0.1, 0.3],  # -log(0) gives infinity
#                             [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08]
#                             ])

# class_targets = [0,1,1]

# print(-np.log(softmax_outputs[[0,1,2],[class_targets]]))
# print(np.mean(-np.log(softmax_outputs[[0,1,2], class_targets])))

import matplotlib.pyplot as plt
import matplotlib
from nnfs.datasets import spiral_data
matplotlib.use('Agg')

np.random.seed(0)



X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases 

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs- np.max(inputs, axis=1, keepdims= True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
    
        return negative_log_likelihoods
    



dense1 = Layer_Dense(2,5)
activation1 = Activation_Relu()

dense2 = Layer_Dense(5,3)
activation2 = Activation_Softmax()

dense1.forward(X)
print('Before  relu Activation')
print(dense1.output)
activation1.forward(dense1.output)
print("After relu Activation")
print(activation1.output)

dense2.forward(activation1.output)
print('Before  softmax Activation')
print(dense2.output)
activation2.forward(dense2.output)
print("After softmax Activation")
print(activation2.output[:5])


loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)