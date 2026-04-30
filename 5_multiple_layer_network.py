import math
'''
Function Name : ReLU
Description : Applies ReLU activation function 
Function parameters : weighted_sum_value i.e Z
Formula : ReLU(z) = max(0 , z)
Use : Commonly used hidden layer
'''
def ReLU(z):
    return max(0,z)

'''
Function Name : Sigmoid
Description : Applies Sigmoid activation function
Formula : sigmoid(z) = 1 / (1+e^-z)
use : commonly used in output layer for binary classification
range : 0 to 1
'''
def Sigmoid(z):
    return 1 / (1 + math.exp(-z))


'''
Function Name : Calculate_weighted_sum()
Description : Calculate weighted sum of inputs
Formula : z = (x1*w1 + x2*w2 + x3*w3 + ......+xn*wn)+bias
Parameter : 
    inputs --> list of input values
    weight --> list of weight values
    bias  --> Bias value
    Return  --> weighted Sum 
'''
def Calculate_weighted_sum(inputs , weights , bias):
    weighted_sum = sum(x*w for x , w in zip(inputs , weights))+bias
    return weighted_sum


'''
Function Name : Display_Multiplication_Details()
Description : Display step by step multiplication of inputs and weights of one neuron
Parameters : 
    inputs -> list of input values
    weights -> list of weight values
'''


def Display_Multiplication_Details(inputs , weights):
    print("Step 1 : Multiplly inputs by corresponding weights")
    for index in range(len(inputs)):
        print(f"{inputs[index]}*{weights[index]} = {inputs[index] * weights[index] : .3f}")
'''
    Function Name : Process_Hidden_Layer()
    Description : Process all neurons of Hidden Layer
                using ReLU activation Function
    Parameters :
        inputs -> Input values from input layer
        hidden_weights -> weight matrix of hidden layer
        hidden_biases -> Bias list of hidden neurons
        Returns : List of hidden Layer outputs 
'''
def Process_Hidden_Layer(inputs , hidden_weights , hidden_biases):
    hidden_outputs = []
    print("=============Hidden Layer=============")
    for neuron_index in range(len(hidden_weights)):
        print(f"Hidden Neuron {neuron_index+1} : ")

        current_weights = hidden_weights[neuron_index]
        current_bias = hidden_biases[neuron_index]

        #Display Multiplication Tables
        Display_Multiplication_Details(inputs , current_weights)
        
        #Calculate Weighted Sum
        z_value = Calculate_weighted_sum(inputs , current_weights , current_bias)

        print(f"Step 2 : Add all multiplication Results and bias{current_bias}")
        print(f"Z = {z_value : .2f}")

        #Apply ReLU activation
        activated_output = ReLU(z_value)
        print(f"Apply ReLU Activation")
        print(f"ReLU({z_value:.3f}) = {activated_output:.2f}\n")

        hidden_outputs.append(activated_output)
    
    return hidden_outputs

'''
Function Name : Process_Output_Layer
Description : process output layer neuron using sigmoid activation function
Parameters :   
    hidden_outputs -> outputs from hidden layer
    output_weights  --> weights of outputs neuron
    output_bias  --> bias of outputs neuron
    return final weighted_sum and final output
'''
def Process_Output_Layer(hidden_outputs , output_weights , output_bias):
    print("=======Output Layer=======")
    print("step 1 : Multiply Hidden layer outputs  by output weights")
    Display_Multiplication_Details(hidden_outputs , output_weights)

    #Calculate Weighted sum for output layer
    z_output = Calculate_weighted_sum(hidden_outputs , output_weights , output_bias)

    print(f"Step 2 : Add all multiplication results and bias {output_bias}")
    print(f" Z = {z_output:.3f}")

    #Apply Sigmoid Function
    final_output = Sigmoid(z_output)
    print("Step 3 : Apply Sigmoid Activation")
    print(f"Sigmoid{z_output:.3f} = {final_output:.3f}")

    return z_output , final_output

'''
function Name : Display_Network_Summary
Description : Display final output of netwrok
Parameters : 
    hidden_outputs : Hidden Layer outputs
    final_output : Output layer final value
'''
def Display_Network_Summary(hidden_outputs , final_output):
    print(f"Hidden Layer output : {hidden_outputs}")
    print(f"Final Network output : {final_output:.3f}")
    print(f" Confidence Percentage : {final_output*100 : .2f}%")

    if final_output >= 0.5:
        print("Prediction :  Positive Class")
    else:   
        print("Pridiction : Negative class")
'''
    Function Name = ANN_Forward_Pass()
    Description : Complete Forward Pass of ANN
    #parameters : 
        inputs -> List of input values
    
    Flow :
        input layer -> Hidden Layer -> output Layer
'''
def ANN_Forward_Pass(inputs):
    print("==== Input Layer ====")
    print(f"Input X1 = {inputs[0]}")
    print(f"Input X2 = {inputs[1]}")

    '''
    Hidden Layer weights and Biases
    Two neurons in hidden layer
    '''
    hidden_weights = [
        [0.5 , -0.2] ,  #weights for hidden neuron 1
        [0.8 , 0.4]     #weights for hidden neuron 2
    ]

    hidden_biases = [
        0.1 , #Bias for hidden neuron 1 
        -0.1  #Bias for hidden neuron 2     
    ]

    '''
    output layer weights and bias
    one neuron in output layer  
    '''

    output_weight = [1.0 , -1.5]
    output_bias = 0.2

    #Process Hidden Layer 
    hidden_outputs = Process_Hidden_Layer(
        inputs , 
        hidden_weights ,
        hidden_biases
    )

    #process output layer
    z_output , final_output = Process_Output_Layer(
        hidden_outputs , 
        output_weight , 
        output_bias
     )

'''
    #main Function
    #purpose : Enter Point function
'''
def main():
    #Example of input
    inputs = [2.0 , 3.0]

    ANN_Forward_Pass(inputs)

#----------------------#
#Starter Function
#----------------------#

if __name__ == "__main__":
    main()