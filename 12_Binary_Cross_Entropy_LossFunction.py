import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
from keras.losses import binary_crossentropy



def BCE(y_actual , y_pred):
    total_loss = 0

    for i in range(len(y_actual)):
        y_true = y_actual[i]
        y_Pred =y_pred[i]

        total_loss = -(y_true * np.log(y_Pred) + (1 - y_true) * np.log(1 - y_Pred)) + total_loss

    return total_loss / len(y_actual)
def main():
    y_true = [1 , 0 , 1]
    y_pred = [0.9 , 0.2 , 0.8]

    total_loss = BCE(y_true , y_pred)
    print(f"Total Loss Using Formula = {total_loss:.2f}")
    
    total_loss = binary_crossentropy(y_true , y_pred)
    print(f"Total loss using in-built function = {total_loss:.2f}")

if __name__ == "__main__":
    main()