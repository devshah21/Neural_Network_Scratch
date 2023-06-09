from activation import Activation
import numpy as np

class Tanh(Activation): #see graph of tanh(x)
    
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
    