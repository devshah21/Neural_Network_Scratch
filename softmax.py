import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]

E = math.e

#exp_vals = []

#for output in layer_outputs:
    #exp_vals.append(E**output)

#norm_base = sum(exp_vals)
#norm_vals = []

#for value in exp_vals:
    #norm_vals.append(value/norm_base)

#print(sum(norm_vals))


## ALT ##

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims = True)

print(norm_values)

# for a 2D matrix, if axis = 0, it's sum of columns
# if axis = 1, sum of rows



#norm_values = exp_values / np.sum(exp_values)
