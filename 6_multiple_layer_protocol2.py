'''
Neural network application with multiple layer
Neural structure : 
    Input layer -> 3 neurons
    Total Hidden Layer 2
        Hidden Layer I --> 3 Neurons
        Hidden Layer II --> 2 Neurons
        [In hidden Layer we use ReLU as a activation Function]
    
    Output Layer
        [In output Layer we use Sigmoid as a activation function] 
'''
import math
def ReLU(z):
    return max(z , 0)

def sigmoid(z) : 
    return 1 / (1+math.exp(-z))

def Process_Hidden_Layer(inputs , weights , bias):
    output = []
    for neuron in range(len(weights)):
        print(f"Neuron Number : {neuron + 1}")

        current_weight = weights[neuron]
        current_bias = bias[neuron]

        weighted_sum = sum(x*w for x , w in zip(inputs , current_weight))+current_bias

        activated_output = ReLU(weighted_sum)
        output.append(activated_output)
    print(output)
    return output

def Process_output_Layer(inputs , weights , bias):
    final_output = 0 
    weighted_sum = sum(x*w for x , w in zip(inputs , weights)) + bias
    final_output = sigmoid(weighted_sum)
    print(f"Final Output = {final_output}")
    return final_output

def network_summary(output) :
    print(f"Fianl Output = {output}")
    if output >= 0.5:
        print("Positive Class")
    else:
        print("Negative Class")

def ANN_Forward_pass(inputs):
    hiddenLayer_I_Weights = [
        [0.1 ,0.2 , 0.3] , 
        [0.2 , -0.4 , 0.1] , 
        [-0.2 , 0.5 , -0.1]
    ]
    hiddenLayer_I_bias = [
        1.0 , 
        -0.1 , 
        1.5
    ]
    hiddenLayer_II_Weights = [
        [0.4 , -0.6 , 0.3] , 
        [0.1 , 0.4 , 0.2]
    ]

    hiddenLayer_II_bias = [
        1.4 , 
        -1.2
    ]

    outputLayer_weight = [
        0.3 , 0.4
    ]

    outputLayer_bias = 1.0

    print("="*60)
    print("Processing Hidden Layer I".center(30))
    print("="*60)

    hiddenLayer_I_output = Process_Hidden_Layer(
        inputs , 
        hiddenLayer_I_Weights , 
        hiddenLayer_I_bias
    )

    print("="*60)
    print("Processing Hidden Layer II".center(30))
    print("="*60)

    hiddenLayer_II_output = Process_Hidden_Layer(
        hiddenLayer_I_output , 
        hiddenLayer_II_Weights , 
        hiddenLayer_II_bias
    )

    print("="*60)
    print("Processing Output Layer".center(30))
    print("="*60)

    output = Process_output_Layer(
        hiddenLayer_II_output ,
        outputLayer_weight , 
        outputLayer_bias
    )

    print("="*60)
    print("Network Summary".center(30))
    print("="*60)
    network_summary(output)

def main():
    inputs = [1.0 , 4.2 , 3.0]
    ANN_Forward_pass(inputs)
if __name__ == "__main__":
    main()
