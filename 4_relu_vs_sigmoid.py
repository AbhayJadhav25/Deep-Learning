import numpy as np
import matplotlib.pyplot as plt
import math
def relu(z):
    '''
    formula : max(0,z)
    '''
    return max(0,z)

def sigmoid(z):
    '''
    formula : 1 / (1 + e^-z)
    '''
    return 1 / (1 + math.exp(-z))

def neuron_forward(inputs , weights , bias , activation):
    print("Step 1 : Calculate weighted Sum")
    weighted_sum = sum(x*w for x , w in zip(inputs , weights)) + bias

    print("Weighted Sum = ",weighted_sum)

    print("Step 2 : Apply Activation function")
    output = activation(weighted_sum)

    print("Activation Function Name : ",activation.__name__)
    print(f"Output = {output : .2f}")

    print("Calculation End")

def reluXsigmoid_plot():
    z_values = np.linspace(-10 , 10 , 200)
    relu_values =  np.maximum(0 , z_values)
    sigmoid_values = 1 / (1 + np.exp(-z_values))

    plt.figure(figsize=(8,5))
    plt.plot(z_values , sigmoid_values , label = "Sigmoid" , linewidth = 2)
    plt.plot(z_values , relu_values , label = "ReLU" , linewidth = 2)

    plt.axhline(y = 0 , linewidth = 0.5)
    plt.axhline(y = 1 , linewidth = 0.5)
    plt.axvline(x = 0 , linestyle = "--")

    plt.title("ReLU vs Sigmoid" , fontsize = 16)
    plt.xlabel("Input" , fontsize = 14)
    plt.ylabel("output" , fontsize = 14)
    plt.grid(True , linestyle = "--" , alpha = 0.6)
    plt.legend()
    plt.show()
def main():
    inputs = [1.0 , 2.0 , 3.0]
    weights = [0.6 , 0.4 , 0.2]
    bias = 0.5
    #Sigmoid
    print("===Sigmoid Activation===")
    neuron_forward(inputs , weights , bias , sigmoid)

    #relu
    print("===Relu Activation===")
    neuron_forward(inputs , weights , bias , relu)
    reluXsigmoid_plot()

if __name__ == "__main__":
    main()