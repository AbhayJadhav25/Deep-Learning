#Neural Network application with Relu Function
import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return max(0,x)

def neuron_forward(input , weights , bias):
    weighted_sum = sum(x*w for x,w in zip(input , weights))+bias

    print("Step 1 : Weighted Sum Calculation")
    print("Weightes_Sum = w.x + b  :  ",weighted_sum)

    output = relu(weighted_sum)
    print("Step 2 : Activation function applied")
    print("Activation Function : ReLU")
    print("Output  : " , output)

    print("====================Calculation End====================")

    return weighted_sum , output


#Plot ReLU function
def plot_relu():
    #Generate range of values for Z
    z_values = np.linspace(-10 , 10 , 200) 
    relu_values = np.maximum(0 , z_values)

    plt.figure(figsize = (8,5))
    plt.plot(z_values , relu_values , label = "ReLU Function" , linewidth = 2 , color = "green")

    #Axes lines

    plt.axhline(y=0 , color = "black" , linewidth = 0.5)
    plt.axvline(x=0 , color = "gray" , linestyle = "--")

    plt.title("ReLU Activation Function" , fontsize = 16)
    plt.xlabel("Inputs " , fontsize = 14)
    plt.ylabel("Outpt" , fontsize = 14)

    plt.grid(True , linestyle = "--", alpha = 0.6)
    plt.legend()

    plt.show()
def main():
    inputs = np.array([1.0 , 2.0 , 3.0])

    weights = np.array([0.6,0.4,0.2])

    bias = 0.5

    neuron_forward(inputs , weights , bias)

    plot_relu()

if __name__=="__main__":
    main()
