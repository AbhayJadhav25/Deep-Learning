'''
steps : 
1 : Read the data from csv
2 : Data analysis
3 : Preprocessing
4 : Train-test split
5 : Feature Scaling
6 : FNN model training
7 : Evaluation
8 : Model saving
9 : Model loading and reusing
10 : Graphical representation
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

'''
Step 1 : Load the Dataset and read it
'''
data = pd.read_csv("placement_data.csv")
print("Data Loaded Successfully")
print("Complete Dataset\n")
print(data)

'''
step 2 : Data Analysis
'''
print(f"Initail 5 Rows of Dataset \n{data.head()} ")
print(f"Dataset Shape = {data.shape}")
print(f"Column Name : \n {data.columns.to_list()}")
print(f"Statistical Summary : \n {data.describe()}")
print(f"Missing Values = {data.isnull().sum()}")

'''
step 3 : Seprate Input features and target output i.e label
'''
x = data.drop('Placed' , axis = 1)
y = data['Placed']

print("Input Values : \n")
print(x.head())

print("Taregt Value:\n")
print(y.head())

'''
step 4 : Train-test split
'''
x_train , x_test , y_train , y_test = train_test_split(
    x ,
    y , 
    test_size= 0.3 , 
    random_state=42
)

print("Training Input Shape : ",x_train.shape)
print("Training Output shape : ",y_train.shape)

print("Testing Input Shape : ",x_test.shape)
print("Testing output shape : ",y_test.shape)

'''
step 5 : Feature Scaling
'''
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Scaled Training Data sample : ")
print(x_train_scaled[:5])

'''
step 6 : FeedForward Neural Network Training
'''
model = MLPClassifier(
    hidden_layer_sizes=(8,4) ,
    activation="relu" ,
    solver='adam' ,
    max_iter=2000,
    random_state=42
)
print(model)

''''
step 7 : Train the model
'''
model.fit(x_train_scaled , y_train)
print("Training Completed")

'''
step 8 : Predict the output
'''
y_pred = model.predict(x_test_scaled)

print("Actual Values : \n")
print(y_test.values)

print("Predicted Values : \n")
print(y_pred)


'''
step 9 : Evaluate Model Performance
'''
accuracy = accuracy_score(y_test , y_pred)
print(f"Accuracy Score = {accuracy:.2f} ")

cm = confusion_matrix(y_test , y_pred)
print("Confusion Matrix : \n")
print(cm)

print("Classification Report \n")
print(classification_report(y_test , y_pred))

'''
step 10 : Predict Probability
model.predict_proba gives model confidence
'''
y_prob = model.predict_proba(x_test_scaled)
print("Prediction Probabilities : \n")
print(y_prob[:5])

'''
step 11 :Saving the train model and scaler
'''
joblib.dump(model , "placement_fnn_model.pkl")
joblib.dump(scaler , "placement_scaler.pkl")

print("Model saved as : placement_fnn_model.pkl")
print("Scaler saved as : placement_scaler.pkl")

'''
step 12 : Load the model and scaler again
'''
loaded_model = joblib.load("placement_fnn_model.pkl")
loaded_scaler = joblib.load("placement_scaler.pkl")

print("Saved model and scaler loaded successfully")

'''
Step 13 :predict for new student
Aptitude : 65
coding : 70
communication : 75
Academics : 70
Internship : 1
'''
new_student = [[65 , 70 , 75 ,70 , 1]]
new_student_scaled = loaded_scaler.transform(new_student)

new_student_pred = loaded_model.predict(new_student_scaled)
new_stud_prob = loaded_model.predict_proba(new_student_scaled)

if new_student_pred[0] == 1:
    print("Prediction : placed")
else : 
    print("Prediction : Not placed")

print("Prediction Probability : \n")
print(new_stud_prob)

# ------------------------------------------------------------
# Step 14: Graph 1 - Placement count
# ------------------------------------------------------------
placement_counts = data["Placed"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(["Not Placed", "Placed"], placement_counts.values)
plt.title("Placement Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Step 15: Graph 2 - Aptitude vs Coding
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

for i in range(len(data)):
    if data["Placed"][i] == 1:
        plt.scatter(data["Aptitude"][i], data["Coding"][i], marker='o', label="Placed" if i == 4 else "")
    else:
        plt.scatter(data["Aptitude"][i], data["Coding"][i], marker='x', label="Not Placed" if i == 0 else "")

plt.title("Aptitude vs Coding")
plt.xlabel("Aptitude Score")
plt.ylabel("Coding Score")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Step 16: Graph 3 - Training Loss Curve
# loss_curve_ stores the loss value after each iteration
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ------------------------------------------------------------
# Step 17: Graph 4 - Actual vs Predicted
# ------------------------------------------------------------
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

plt.figure(figsize=(8, 5))
plt.plot(comparison["Actual"].values, marker='o', label="Actual")
plt.plot(comparison["Predicted"].values, marker='s', label="Predicted")
plt.title("Actual vs Predicted Placement")
plt.xlabel("Test Sample Index")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.show()

print("\nProject execution completed successfully")