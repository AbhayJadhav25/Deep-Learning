'''
.ravel ---> it is helpful to convert 2D array into 1D flattended array.
it is a function belongs to the numpy Arrays
'''
import numpy as np

y = np.array([[1] , [2] ,[3] , [4] , [5]])
print(y)
print("Y shape = ",y.shape) #(5,1)

y = y.ravel()
print(y)
print("Shape of y = ",y.shape)

'''
output : 
    [1 2 3 4 5]
Shape of y =  (5,)
'''