'''
Student Pass Fail prediction using feedForward network
Features : Study Hours , Attendance , Assignment score
Label : 0 = Fail , 1 = Pass
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

'''
Step 1 : Each row Dataset Contains Features i.e X
output : 0 = Fail , 1 = Pass
'''

x = [
    [1, 40, 30],
    [2, 50, 35],
    [3, 60, 40],
    [4, 65, 50],
    [5, 70, 55],
    [6, 75, 65],
    [7, 80, 70],
    [2, 45, 25],
    [8, 90, 85],
    [1, 35, 20],
    [3, 55, 45],
    [4, 68, 52],
    [5, 72, 58],
    [6, 78, 62],
    [7, 85, 75]
]

y = [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]

'''
Step 2 : Split the dataset into training and testing data
Training Data --> used to teach the model
Testing Data --> Used to check performance
'''

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=42 , stratify=y)

'''
step 3 : Scaled X_train , X_test 
'''
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
'''
step 4 : Create the FNN 
Hidden Layer Size = (5 ,) --> one hidden layer with 5 neurons
activation = 'relu
max_iter = 1000 means training runs upto 1000 times if needed
'''

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    activation = 'relu' ,
    solver = 'lbfgs' , 
    max_iter=1000 ,
    random_state=42
)

''' 
step 5 : Train the model
Model learns patterns from training data
'''
model.fit(x_train_scaled , y_train)

'''
Step 6 : predict the test data
'''
prediction = model.predict(x_test_scaled)

'''
Step 7 : Check model accuracy
'''
print("Actual Output : ",y_test)
print("Predicted Output : ",prediction.tolist())

accuracy = accuracy_score(y_test , prediction)
print(f"Accuracy : {accuracy*100}")

#detail report 
print("\nClassification Report : ")
print(classification_report(y_test , prediction))

'''
step 8 : test with new student data
Example : Study hours = 5 , Attendance = 75 , Assignment_score = 60
'''
new_student = [[5,75,60]]
new_student_scaled = scaler.transform(new_student)
prediction = model.predict(new_student_scaled)

if prediction[0] == 1:
    print("\nNew Student Prediction : Pass")
else:
    print("\nNew Student prediction is Fail")