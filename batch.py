import numpy as np

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


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense()

