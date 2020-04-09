import numpy as np
from load_data import *
import gmpy2
from gmpy2 import mpz

data_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'

TOTAL_SIZE = 60000  # TOTAL_SIZE = TRAIN_SIZE + TEST_SIZE
TRAIN_SIZE = 50000
TEST_SIZE = 10000

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.1

np.random.seed(1)

####################################Functions#############################################

def xavier_init(fan_in, fan_out):
    low = -np.sqrt(6.0 / (fan_in + fan_out))
    high = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low, high, (fan_in, fan_out))

def relu(x):    # ReLUpython
    return np.maximum(x, 0)

def drelu(x):   # derivative of ReLU
    return 0 if x <= 0 else 1
drelu = np.vectorize(drelu)


####################################Load data and weight initiation#############################################

X_TOTAL, Y_TOTAL = load_data(data_file, label_file, TOTAL_SIZE)
x_train_total = X_TOTAL[0:TRAIN_SIZE]
y_train_total = Y_TOTAL[0:TRAIN_SIZE]
x_test_total = X_TOTAL[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]
y_test_total = Y_TOTAL[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]

W2 = xavier_init(128, 784)
W3 = xavier_init(10, 128)
b2 = np.zeros((128, 1))
b3 = np.zeros((10, 1))


####################################Training#################################################################
for epoch in range(EPOCHS):
    sum_of_training_error = 0
    sum_of_test_error = 0
    for i in range(int(TRAIN_SIZE/BATCH_SIZE)):
        # LOAD DATA
        x_train = x_train_total[i*BATCH_SIZE:(i+1)*BATCH_SIZE, 0:784]
        y_train = y_train_total[i*BATCH_SIZE:(i+1)*BATCH_SIZE, 0:10]

        # Forward propagation
        a1 = x_train.transpose()                # (784,64)
        z2 = np.dot(W2, a1) % Q + b2            # (128,784) * (784,64)
        a2 = relu(z2)                           # (128,64)
        z3 = np.dot(W3, a2) + b3                # (10,128) * (128,64)
        a3 = relu(z3)                           # (10,64)

        # Backward propagation
        delta_3 = (a3 - y_train.transpose())
        delta_2 = drelu(z2) * np.dot(W3.transpose(), delta_3)

        gradient_b3 = delta_3
        gradient_b2 = delta_2
        gradient_W3 = np.dot(delta_3, a2.transpose())
        gradient_W2 = np.dot(delta_2, a1.transpose())

        sum_gradient_b3 = np.sum(gradient_b3, axis=1).reshape(-1, 1)
        sum_gradient_b2 = np.sum(gradient_b2, axis=1).reshape(-1, 1)
        sum_gradient_W3 = gradient_W3
        sum_gradient_W2 = gradient_W2

        b3 -= LR * sum_gradient_b3 / BATCH_SIZE
        b2 -= LR * sum_gradient_b2 / BATCH_SIZE
        W3 -= LR * sum_gradient_W3 / BATCH_SIZE
        W2 -= LR * sum_gradient_W2 / BATCH_SIZE

        # Training Error
        sum_of_training_error += np.sum(np.argmax(a3, axis=0) != np.argmax(y_train.transpose(), axis=0)).astype(int)

    test_layer_2 = relu(np.dot(W2, x_test_total.transpose()) + b2)
    test_layer_3 = relu(np.dot(W3, test_layer_2) + b3)
    sum_of_test_error = np.sum(np.argmax(test_layer_3, axis=0) == np.argmax(y_test_total.transpose(), axis=0)).astype(int)
    print("epoch:", epoch, "training_error:", sum_of_training_error, "accuracy_rate:", sum_of_test_error/TEST_SIZE)















