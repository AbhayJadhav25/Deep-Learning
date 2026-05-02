'''
Training Process Demo
'''

#import random to randomly initialize the weight
import random
import matplotlib.pyplot as plt
print("="*70)
print("Step 1 : Initial Data".center(70))
print("="*70)

x = 2
actual_output = 10

print(f"Input Value = {x}")
print(f"Actual Output = {actual_output}")

print("="*70)
print("Step 2 : Randomly initialize weight".center(70))
print("="*70)

weight = random.uniform(0,1) #random value between 0 & 1
learning_rate = 0.1

print(f"Initial Weight = {weight}")
print(f"Learning rate = {learning_rate}")

print("="*70)
print("Training Started".center(70))
print("="*70)

steps = []
predicted_list = []
weight_list = []
for step in range(1,13):

    print("="*70)
    print(f"Step {step}".center(70))
    print("="*70)

    predicted_output = weight * 2
    error = actual_output - predicted_output
    loss = error**2
    weight = weight + (learning_rate * error * x)

    print(f"Predicted Output = {predicted_output}")
    print(f"Error = {error}")
    print(f"Loss = {loss}")
    print(f"Updated weight = {weight}")

    steps.append(step)
    predicted_list.append(predicted_output)
    weight_list.append(weight)

print("="*70)
print("Final Output".center(70))
print("="*70)

output = x * weight

print(f"Final Weight = {weight}")
print(f"Final Prediction = {output}")
print(f"Expected Output = {actual_output}")

#Graph 1 : Prediction vs Actual
actual_line = [actual_output] *len(steps)
plt.figure()
plt.plot(steps , predicted_list , marker = "o" , label = "Predicted Output")
plt.plot(steps , actual_line , linestyle = "--" , label = "Actual Output")
plt.title("Prediction Approaching Actual output")
plt.xlabel("Steps")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

#Graph 2 : Weight Change
plt.figure()
plt.plot(steps , weight_list , marker = "o")
plt.title("weight Adjustment during learning")
plt.xlabel("Steps")
plt.ylabel("Weight Value")
plt.grid()
plt.show()
