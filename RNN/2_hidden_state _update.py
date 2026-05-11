#This code demonstrate how hidden state update

import numpy as np

sequence = [1,2,3]  #I = 1 , Love = 2 , AI = 3

Wx , Wh , b = 0.5 , 0.8 , 0.1  #Initialize weights , hidden state , and bias
h = 0  #initial hidden state

print("========================================================")
print("Processing Sequence Step by step".center(40))
print("========================================================")

for t , x in enumerate(sequence):
    h = np.tanh(Wx*x + Wh*h + b)
    print(f"Timestep = {t+1} | Input = {x} | Hidden State = {h:.2f}")