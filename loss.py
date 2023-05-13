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