#RNN example with keras to demonstrate sequential input

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN , Dense , Embedding

X = np.random.randint(20 , size = (10,5))
Y = np.random.randint(2 , size = (10,1))
print(X)
print(Y)
model = Sequential()
model.add(Embedding(input_dim=20 , output_dim=8 , input_length=5))
model.add(SimpleRNN(16 , activation = 'tanh'))
model.add(Dense(1 , activation = 'sigmoid'))

#compile model
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#train
model.fit(X , Y , epochs = 5 , verbose = 1)

print("Sample prediction : ",model.predict(X[:1]))
