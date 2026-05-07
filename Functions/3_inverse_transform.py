'''
To convert scaled values back to the original values
'''
from sklearn.preprocessing import StandardScaler
import numpy as np

y = [22000 , 26000 , 40000]
scaler = StandardScaler()

# y = scaler.fit_transform(y) 
# print(y) #ValueError: Expected 2D array, got 1D array instead:

y = np.array(y).reshape(-1,1)
y_scaled = scaler.fit_transform(y)
print("Y Scaled Value")
print(y_scaled)

original = scaler.inverse_transform(y_scaled)
print("Original Value")
print(original.ravel())