#Neural Network Application with Sigmoid Function
import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
        formula : 1 / (1 + e^x)
    """
    return 1/(1 + math.exp(-x))

def neuron_forward(inputs , weights , bias):
    #step 1 : Calculate Weighted Sum i.e Z

    z = sum(x*w for x , w in zip(inputs , weights))+bias 

    print("Step 1 : Weighted Sum Calculation ")
    print("Weighted Sum  =  " , z)

    print("Step 2 : Applied Activate Function")
    print("Activate Function : Sigmoid Function")
    output = sigmoid(z)
    print(f"Outpt  : {output :.2f}")
    print("=============Calculation Ended===============")

def plot_sigmoid():
    z_values = np.linspace(-10 , 10 , 200)
    sigmoid_values = 1 / (1 + np.exp(-z_values))

    plt.figure(figsize=(8,5))
    plt.plot(z_values , sigmoid_values , label = "Sigmoid Function" , linewidth = 2 , color ="green")

    plt.axhline(y = 0 , color = "black" ,linewidth = 0.5)
    plt.axvline(x = 0 , color = "gray" ,linestyle = "--")
    plt.axhline(y = 1 , color = "black" ,linewidth = 0.5)

    plt.title("Activation Function (Sigmoid Function)" , fontsize = 16)
    plt.xlabel("Input(z)" , fontsize = 14)
    plt.xlabel("Output(probability)" , fontsize = 14)

    plt.grid(True , linestyle = "--" , alpha = 0.6)
    plt.legend()
    plt.show()
def main():
    inputs = np.array([1.0 , 2.0 , 3.0])
    weights = np.array([0.6 , 0.4 , 0.2])
    bias = 0.5

    neuron_forward(inputs , weights , bias)
    plot_sigmoid()
if __name__ == "__main__":
    main()