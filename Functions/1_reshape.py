'''
Convert 1D array into 2D array.
Some scikit learn tools like StandardScaler() does not work well on 1D array , they expected 2D array
'''
from sklearn.preprocessing import StandardScaler
import numpy as np

y = [22000 , 26000 , 40000]
scaler = StandardScaler()

# y = scaler.fit_transform(y) 
# print(y) #ValueError: Expected 2D array, got 1D array instead:

y = np.array(y).reshape(-1,1)
y = scaler.fit_transform(y)
print(y)