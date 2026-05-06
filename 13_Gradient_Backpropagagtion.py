'''
Demonstrate :
Forward Pass
Backpropagation
Chain rule
Gradient Descent and weight , bias update
'''
import math
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def derivate_sigmoid(output):
    return output * (1 - output)

#input neuron

x1 = 1.0 
x2 = 2.0

#target value
target = 1.0

'''
Initialize weights , bias and learning rate
'''
w1 = 0.5
w2 = -0.3 
b = 0.1
learning_rate = 0.1

print("Initial Value")
print(f"w1 = {w1:.2f}")
print(f"w2 = {w2:.2f}")
print(f"Bias (b) = {b:.2f}")
print(f"Learning rate  =  {learning_rate:.2f}")


for epochs in range(0 , 10):
    print(f"=================== Epochs {epochs+1} ===================")

    z = (w1*x1) + (w2*x2) + b
    output = sigmoid(z)

    #Here we use Mean Squared Error
    loss = 0.5 * ((target - output)**2)

    '''
    Back Propagation Step 1 : Derivate of loss with respect to output
    DL / Doutout = output - target 
    '''
    dl_doutout = output - target
    
    '''
    Back Propagation Step 2 : Derivate of output with respect to derivate of z i.e derivate of sigmoid(output)
    Doutput / Dz = sigmoid'(output)
                 = output * (1 - output)
    '''
    doutput_dz = derivate_sigmoid(output)
    
    '''
    Back Propagation Step 3 : Chain Rule
    derivate of loss with respect to derivate of z 
    DL / Dz = (DL / Doutpt) + (Doutput / Dz)
    '''
    dl_dz = dl_doutout + doutput_dz

    '''
    step 4 : Gradient with respect to weight and bias
    z = x1*w1 + x2*w2 + b
    dz_dw1 = w1
    dz_dw2 = w2
    dz_db = 1 (always 1)
    '''
    dl_dw1 = dl_dz * x1
    dl_dw2 = dl_dz * x2
    dl_db = dl_dz

    '''
    step 5 : Gradient Descent update weight and bias
    '''
    w1 = w1 - (learning_rate * dl_dw1)
    w2 = w2 - (learning_rate * dl_dw2)
    b = b - (learning_rate * dl_db)

    print(f"Z = {z:.4f}")
    print(f"Output = {output:.2f}")
    print(f"Loss = {loss:.2f}")
    print(f"dl_doutput  = {dl_doutout:.2f}")
    print(f"doutput_dz = {doutput_dz:.2f}")
    print(f"dl_dz  = {dl_dz:2f}")
    print(f"dl_dw1 = {dl_dw1:.2f}")
    print(f"dl_dw2 = {dl_dw2:.2f}")
    print(f"dl_db =  {dl_db:.2f}")
    print(f"========== Updated weight After {epochs+1}==========\n")
    print(f"w1 = {w1:.2f}")
    print(f"w2 = {w2:.2f}")
    print(f"Bias (b) = {b:.2f}")

print("\n====================== Final Summary ======================\n")
print(f"Final w1 = {w1:.2f}")
print(f"Final w2 = {w2:.2f}")
print(f"Final Bias (b1) = {b:.2f}")
