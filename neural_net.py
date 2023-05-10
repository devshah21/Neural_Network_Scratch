import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

nnfs.init()

X = [[1, 2, 3, 2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

E = math.e
exp_vals = []

norm_base = sum(exp_vals)


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


layer1 = Layer_Dense(2, 5)

activation1 = Activation_Relu()

layer1.forward(X)
activation1.forward(layer1.output)

print(activation1.output)
