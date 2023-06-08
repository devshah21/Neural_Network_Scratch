import numpy as np
import nnfs
from nnfs.datasets import spiral_data

inputs = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0], 
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]
            
bias = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + bias

# we need to convert to a numpy array and then transpose it to make the shapes line up
# we're doing inputs * weights and since we're doing rows x columns, in the input row there are
# 4 elements, but the column for the weights is only 3, hence why we have to transpose the
# weights to make sure the shapes line up

# equivalent to the output variable from main.py

weights1 = [[0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
            
bias1 = [-1, 2, -0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + bias

layer2_output = np.dot(layer1_output, np.array(weights1).T) + bias1

# in general, output = weight * input + bias
# similar to y = mx + b

X = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)


### ACTUAL IMPLEMENTATION ###


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # creates a numpy array of size inputs x neurons with random vals and sized down by mult. by 0.10
        self.biases = np.zeros((1, n_neurons))
        # create a numpy array of 0s as this is the starting point for the biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        # y = mx + b code from earlier

class Activation_Relu:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # check activation.py, but the logic is, if something is < 0, we append 0, else we append the value itself
















layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

# sigmoid functions are used as activation as they are more reliaible to calculate loss

