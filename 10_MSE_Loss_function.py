import numpy as np
from sklearn.metrics import mean_squared_error

def MSE(y_actual , y_pred):
    total_loss = 0

    for i in range(len(y_actual)):
        total_loss = ((y_actual[i] - y_pred[i]) ** 2) + total_loss

    return total_loss / len(y_actual)
def main():
    y_true = [1 , 0 , 1]
    y_pred = [0.9 , 0.2 , 0.8]

    total_loss = MSE(y_true , y_pred)
    print(f"Total Loss Using Formula = {total_loss:.2f}")
    
    total_loss = mean_squared_error(y_true , y_pred)
    print(f"Total loss using in-built function = {total_loss:.2f}")

if __name__ == "__main__":
    main()