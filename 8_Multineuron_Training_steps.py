'''
Multineuron Training Steps Demo
'''
import random

border = "="*70

print(border)
print("Step 1 : Initial Data".center(70))
print(border)

x1 = 2.0
x2 = 3.0
actual_output = 8.0

print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"Actual Output = {actual_output}")

print(border)
print("Step 2 : Initialize random weight and bias")

#Hidden neuron 1 
w11 = random.uniform(0,1)
w12 = random.uniform(0,1)
b1 = random.uniform(0,1)

#Hidden neuron 2
w21 = random.uniform(0,1)
w22 = random.uniform(0,1)
b2 = random.uniform(0,1)

#output neuron
w_out1 = random.uniform(0,1)
w_out2 = random.uniform(0,1)
b_out = random.uniform(0,1)

learning_rate = 0.01

print(border)
print("Initial Parameters".center(70))
print(border)

print("Hidden Layer Neuron - I")
print(f"w11 = {w11:.4f}\tw12 = {w12:.4f}\tBias(b1) = {b1:.4f}")

print("Hidden Layer Neuron - II")
print(f"w11 = {w21:.4f}\tw12 = {w22:.4f}\tBias(b2) = {b2:.4f}")

print("Output Layer Neuron")
print(f"w_out1 = {w_out1:.4f}\tw_out2 = {w_out2:.4f}\tBias(b_out) = {b_out:.4f}")


print(border)
print("Step 3 : Training Loop".center(70))
print(border)

for step in range(1,51):
    print(border)
    print(f"Step {step}".center(70))
    print(border)

    #-------------------------------------------
    #Forward Passs Hidden Neuron - I
    #-------------------------------------------

    z1 = x1*w11 + x2*w12 + b1
    h1 = max(0,z1)

    print(" ================ HHidden Neuron - I ================\n")

    print(f"z1 = ({x1} * {w11:.4f}) +({x2} * {w12:.4f})+ {b1:.4f}) = {z1:.4f}")

    print(f"h1 = relu(z1) = {h1:.4f}")

    #-------------------------------------------
    #Forward Passs Hidden Neuron - II
    #-------------------------------------------

    z2 = x1 * w21 + x2 * w22 + b2
    h2 = max(0,z2)

    print(" ================ HHidden Neuron - I ================\n")

    print(f"z1 = ({x1} * {w21:.4f}) +({x2} * {w22:.4f})+ {b2:.4f}) = {z2:.4f}")

    print(f"h2 = relu(z2) = {h2:.4f}")

    #-------------------------------------------
    #Forward Passs Output Neuron 
    #-------------------------------------------

    predicted_output = (h1 * w_out1) + (h2*w_out2) + b_out

    print("================ Output Neuron ================\n")

    print(f"Predicted Output = ({h1:.4f} * {w_out1:.4f}) + ({h2:.4f} * {w_out2:.4f}) + {b_out:.4f} = {predicted_output:.4f}")
    print(f"Predicted Output  = {predicted_output:.4f}")

    error = actual_output - predicted_output
    loss = error **2

    print(f"Total Error = {error}")
    print(f"Total Loss = {loss}")

    w_out1 = w_out1 + (learning_rate * error * h1)
    w_out2 = w_out2 + (learning_rate * error * h2)
    b_out  = b_out + (learning_rate * error)

    if z1>0:
        w11 = w11 + (learning_rate*error*x1*w_out1*0.1)
        w12 = w12 + (learning_rate*error*x2*w_out1*0.1)
        b1 = b1 + (learning_rate*error*w_out1*0.1)

    if z2 > 0:
        w21 = w21 + (learning_rate * error * x1 * w_out2 * 0.1)
        w22 = w22 + (learning_rate * error * x2 *w_out2 * 0.1)
        b2 = b2 + (learning_rate * error * w_out2 * 0.1)

    print("================ Updated Parameter ================\n")
    print("Hidden Neuron I\n")
    print(f"w11 = {w11:.2f}\tw12 = {w12:.2f}\tBias(b1) = {b1:.2f}")

    print("Hidden Neuron II")
    print(f"w21 = {w21:.2f}\tw22 = {w22:.2f} \t Bias(b2) = {b2:.2f}")

    print("Output Neuron")
    print(f"w_out1 = {w_out1:.2f}\tw_out2 = {w_out2:.2f}\t bias_out  = {b_out:.2f}")

print(border)
print("Final Result".center(70))
print(border)

final_z1 = (x1 * w11) + (x2 * w12) + b1
fianl_h1 = max(0 , final_z1)

final_z2 = (x1 * w21) + (x2 * w22) + b2
final_h2 = max(0,final_z2)

final_output = (fianl_h1 * w_out1) + (final_h2 * w_out2) + b_out

print(f"Final Hidden Output h1 = {fianl_h1:.2f}")
print(f"Final Hidden Output h2 = {final_h2:.2f}")
print(f"Final Predicted Output = {final_output:.2f}")
print(f"Expected Output = {actual_output:.2f}")
print(f"Final Error  = {actual_output - predicted_output :.2f}")