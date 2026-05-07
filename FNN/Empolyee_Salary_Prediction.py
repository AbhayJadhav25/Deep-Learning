'''
Empolyee Salary Prediction Case study
Description : Predict Empolyee salary using Feedforward Neural Network
Features : Experience , Education score , Skill rating , certification
Label : salary
'''
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

#Dataset

x = [
    [1,5,4,0] , 
    [2,6,5,1],
    [3,6,6,1],
    [4,7,7,2],
    [5,7,8,2],
    [6,8,8,3],
    [7,8,9,3],
    [8,9,9,4],
    [10,9,10,5],
    [9,9,10,4]
]

#Salary output
y = [22000,26000,32000,40000,47000,
     54000,62000,70000,85000,78000]

y = np.array(y).reshape(-1,1)  #for scaling covert y into 2D form

#split Data
x_train , x_test , y_train, y_test = train_test_split(
    x , y , train_size=0.75 , random_state=42 
) 

#scaling Input and Output
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

y_train_scaled = scaler_y.fit_transform(y_train).ravel()
y_test_scaled = scaler_y.transform(y_test).ravel()

#create FNN Model

model = MLPRegressor(
    hidden_layer_sizes=(6,) ,
    activation="relu" , 
    solver = "lbfgs" , 
    max_iter=5000,
    random_state=42
)


#train_model

model.fit(x_train_scaled , y_train_scaled)

#predict on test data
pre_scaled = model.predict(x_test_scaled)
print(pre_scaled)

# #convert predictions back to the original salary range
# predictions = scaler_y.inverse_transform(pre_scaled.reshape(-1,1)).ravel() #reshape here because .inverse_transform needs 2D array as input. and after tranformation we again convert these 2D array into 1D array.

# print("Actual Salaries : ",y_test.ravel())
# print("Prediction Salary : ",predictions.astype(int)) #convert float salries into integer

# #error
# error = mean_absolute_error(y_test.ravel() , predictions)
# print("Average Error : ",error)

# '''
# New Empolyee Prediction
# Experience = 5 years
# Education = 8
# Skill = 9
# Certification = 3
# '''
# new_emp = [[5,8,9,3]]
# new_emp_scaled = scaler_x.transform(new_emp)

# salary_scaled = model.predict(new_emp_scaled)
# salary = scaler_y.inverse_transform(salary_scaled.reshape(-1,1)).ravel()

# print(f"\nPredicted Salary for new Empolyee : ",int(salary[0]))