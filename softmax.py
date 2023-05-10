import math

layer_outputs = [4,8, 1.21, 2.385]

E = math.e

exp_vals = []

for output in layer_outputs:
    exp_vals.append(E**output)

norm_base = sum(exp_vals)
norm_vals = []

for value in exp_vals:
    norm_vals.append(value/norm_base)

print(sum(norm_vals))