x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

xw = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

z = xw + xw1 +xw2


y = max(0, z) #applying relu activation

dvalue = 1.0

rulu_dz = (1. if z > 0 else 0.)

drulu_dz = dvalue * rulu_dz #compute derivative of ReLU and the chain rule

