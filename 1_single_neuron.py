#Neural network with single neuron

#Network Application with Single neuron 

import numpy as np

inputs = np.array([2.0 , 3.0 , 4.0])
weights = np.array([0.5 , 0.3 , 0.2])

bias = 1.0

weighted_sum = np.dot(inputs , weights) + bias

def relu(x):
    return max(0,x)

output = relu(weighted_sum)

print("=================Final Result=====================")
print("Inputs :  ",input)
print("Weights :  ",weights)
print("Bias  :  ",bias)
print("Weighted Sum :  ",weighted_sum)
print("Final Output  :  ",output)
print("=================Final Result=====================")