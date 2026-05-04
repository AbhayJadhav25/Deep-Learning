import math
import random

def header(str):
    print(f"{"="*70}")
    print(str)
    print(f"{"="*70}")

x1 = 1.5
x2 = 3.2

actual_output = 9.0

#Neuron 1 Weights and Bias
w11 = random.uniform(0,1)
w12 = random.uniform(0,1)
b1 = random.uniform(0,1)

#Neuron 2 weights and bias
w21 = random.uniform(0,1)
w22 = random.uniform(0,1)
b2 = random.uniform(0,1)

#Output Neuron weight and bias
w_out1 = random.uniform(0,1)
w_out2 = random.uniform(0,1)
b_out  = random.uniform(0,1)

learning_rate = 0.02

print(f"{"="*20}Initial Parameter{"="*20}")
print(f"Neuron 1 \n w11 = {w11:.2f}\tw12 = {w12:.2f}\tb1 = {b1:.2f}")
print(f"Neuron 2 \n w21 = {w21:.2f}\tw22 = {w22:.2f}\tb2 = {b2:.2f}")
print(f"Output Neuron \n w_out1 = {w_out1:.2f}\tw_out2 = {w_out2:.2f}\tBias = {b_out:.2f}")

for i in range(0 , 18):
    header(f"{"="*20} Step : {i+1} {"="*20}")

    z1 = (x1 * w11) + (x2 * w12) + b1
    h1 = max(0 , z1)

    z2 = (x1 * w21) + (x2 * w22) + b2
    h2 = max(0,z2)

    print(f"z1 = {z1:.2f}\th1 = {h1:.2f}")
    print(f"z2 = {z2:.2f}\th2 = {h2:.2f}")

    predicted_output = (h1 * w_out1) + (h2 * w_out2) + b_out
    print(f"Predicted Output = {predicted_output:.2f}")

    error = actual_output - predicted_output
    loss = error**2

    print(f"Error = {error:.2f}")
    print(f"Loss = {loss:.2f}")

    w_out1 = w_out1 + (learning_rate * error * h1)   #here we take h1 because h1 is a forward to the output neuron.not x1
    w_out2 = w_out2 + (learning_rate * error * h2)
    b_out = b_out + (learning_rate*error)

    if z1 > 0:
        w11 = w11 + (learning_rate * error * x1 * w_out1 * 0.1)
        w12 = w12 + (learning_rate * error * x2 * w_out1 * 0.1)
        b1 = b1 + (learning_rate * error * w_out1 * 0.1)

    if z2 > 0:
        w21 = w21 + (learning_rate * error * x1 * w_out2 * 0.1)
        w22 = w22 + (learning_rate * error * x2 * w_out2 * 0.1)
        b2 = b2 + (learning_rate * error * w_out2 * 0.1)

    print(f"{"="*10}Update Parameter{"="*10}")
    print("============== Neuron 1 ==============\n")
    print(f"w11 = {w11:.2f}\tw12 = {w12:.2f}\tb1(bias) = {b1:.2f}")
    print("============== Neuron 2 ==============\n")
    print(f"w11 = {w21:.2f}\tw12 = {w22:.2f}\tb2(bias) = {b2:.2f}")
    print("============== Output Neuron ==============\n")
    print(f"w_out1 = {w_out1:.2f}\tw_out2 = {w_out2:.2f}")


final_z1 = (x1 * w11) + (x2 * w12) + b1
final_h1 = max(0,final_z1)
final_z2 = (x1 * w21) + (x2 * w22) + b2
final_h2 = max(0,final_z2)

print(f"Final Z1 = {final_z1:.2f}\tFinal h1 = {final_h1:.2f}")
print(f"Final Z2 = {final_z2:.2f}\tFinal h2 = {final_h2:.2f}")

final_Predicted_output = (final_h1 * w_out1) + (final_h2 * w_out2) + b_out
final_error = actual_output - final_Predicted_output
final_loss = final_error ** 2

print(f"Final Predicted Output = {final_Predicted_output:2f}")
print(f"Final Error = {final_error:.2f}")
print(f"Final Loss  = {final_loss:.2f}")
print(f"Actual Output = {actual_output:.2f}")