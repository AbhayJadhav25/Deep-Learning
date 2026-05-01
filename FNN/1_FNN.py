'''
Simple FNN implementation
Network Structure : 
    input layer 
    Total hidden layer = 1 (Activated Function --> ReLU)
    Output Layer    (Activated Function --> Sigmoid)
'''
import math
'''
Function Name : FNN_Implementation
Description --> Manual Implementation of FNN .
Purpose -> Just show how FNN calculation works.
Parameters
    inputs -> list of inputs
'''
'''
Function : ReLU 
Formula : ReLU(z) = max(0 , z)
use : Activaton function for hidden layer
'''
def relu(z):
    return max(0 , z)


'''
Function : sigmoid
Formula : sigmoid(z) = 1 / (1 + e^-z)
use : Activaton function for output layer
'''
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

'''
Function Name : HiddenLayer_Calculation
Description : Calculation of Hidden Layer
Parameters : 
    inputs --> List of input 
    weights -> List of weight
    bias --> bias of neuron

return : outputs List(which is value of hidden layer neuron)
'''
def HiddenLayer_Calculation(inputs , weights , bias):
    output = []
    z1 = inputs[0] * weights[0][0] + inputs[1]*weights[0][1] + bias[0]
    a1 = relu(z1)
    output.append(a1)

    z2 = inputs[0] * weights[1][0] + inputs[1]*weights[1][1]+bias[1]
    a2 = relu(z2)
    output.append(a2)

    return output

'''
Function Name : OutputLayer_Calculation
Description : Calculation of output Layer
Parameters : 
    inputs --> List of input 
    weights -> List of weight
    bias --> bias of neuron

return : Fianl Output 
'''
def OutputLayer_Calculation(inputs , weights , bias):
    output = inputs[0]*weights[0] + inputs[1]*weights[1] + bias
    print(output)
    final_output = sigmoid(output)
    print(f"Final Output = {final_output}")
    return final_output


def FNN_Implementation(inputs) :
    hiddenLayer_Weights = [
        [0.5 , -0.2] ,
        [0.8 , 0.4]
    ]
    hiddenLayer_bias = [
        0.1 , 
        -0.1
    ]
    outputLayer_weights = [
        1.2 , -0.7
    ]
    outputLayer_bias = 0.05

    #Function Calling

    output = HiddenLayer_Calculation(
        inputs , 
        hiddenLayer_Weights , 
        hiddenLayer_bias
    )

    final_output = OutputLayer_Calculation(
        output , 
        outputLayer_weights , 
        outputLayer_bias
    )

def main():
    inputs = [2.0 , 3.0]
    FNN_Implementation(inputs)
if __name__ == "__main__":
    main()