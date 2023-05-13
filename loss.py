import numpy as np
import math

b = 5.2

print(np.log(b))

print(math.e**np.log(b))


## SECTION ##


softmax_out = [0.7, 0.1, 0.2]

target_class = 0

one_hot_encoding = [1,0,0]

loss = -(math.log(softmax_out[0])*one_hot_encoding[0] + 
                math.log(softmax_out[1])*one_hot_encoding[1] +
                math.log(softmax_out[2])*one_hot_encoding[2])

print(loss)

soft_outputs = np.array([[0.7, 0.1, 0.2], 
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]])

class_targs = [0,1,1]

#print(soft_outputs[[0, 1, 2], class_targs])

print(soft_outputs[range(len(soft_outputs)), class_targs])

new_loss = -np.log(soft_outputs[range(len(soft_outputs)), class_targs])

avg_loss = np.mean(new_loss)

print(new_loss)

# this isn't sufficient as np.mean of 0 is inf
