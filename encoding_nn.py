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
EPOCHS = 1
LR = 0.1

BASE = 10
PRECISION_INTEGRAL = 8
PRECISION_FRACTIONAL = 8
PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
Q = mpz(293973345475167247070445277780365744413)

np.random.seed(1)

####################################Functions#############################################

def encode(rational):
    upscaled = int(rational * BASE**PRECISION_FRACTIONAL)
    field_element = mpz(upscaled) % Q
    return field_element
encode = np.vectorize(encode)

def encode_b(rational):
    upscaled = int(rational * BASE**PRECISION_FRACTIONAL)
    field_element = mpz(upscaled) % Q
    return field_element

def decode(field_element):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = float(upscaled / (BASE**PRECISION_FRACTIONAL))
    return rational
decode = np.vectorize(decode)


def TruncPrMul(x):
    x_b = (x + BASE ** (2 * PRECISION)) % Q
    x_r = x_b % (BASE ** PRECISION_FRACTIONAL)
    x_p = x - x_r
    x_d = (x_p * gmpy2.invert(BASE ** PRECISION_FRACTIONAL, Q)) % Q
    return x_d
TruncPrMul = np.vectorize(TruncPrMul)


def div_enc(a, b):
    x = encode_b(1 / b)
    y = TruncPrMul(a * x % Q)
    return y

def add_enc(a, b):
    return (a + b) % Q
add_enc = np.vectorize(add_enc)

def sub_enc(a, b):
    return (a - b) % Q
sub_enc = np.vectorize(sub_enc)

def mul_enc(a, b):
    return (a * b) % Q
mul_enc = np.vectorize(mul_enc)


def xavier_init(fan_in, fan_out):
    low = -np.sqrt(6.0 / (fan_in + fan_out))
    high = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(low, high, (fan_in, fan_out))


def relu_enc(x):
    return x if x <= int((Q - 1)/2) else 0
relu_enc = np.vectorize(relu_enc)

def drelu_enc(x):
    return 1 if x <= int((Q - 1)/2) else 0
drelu_enc = np.vectorize(drelu_enc)



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

W2_enc = encode(W2)
W3_enc = encode(W3)
b2_enc = encode(b2)
b3_enc = encode(b3)

LR_enc = encode(LR)

####################################Training#################################################################
for epoch in range(EPOCHS):
    sum_of_training_error = 0
    sum_of_test_error = 0
    for i in range(int(TRAIN_SIZE / BATCH_SIZE)):
        # Load and encode data
        x_train = x_train_total[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 0:784]
        y_train = y_train_total[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, 0:10]
        x_train_enc = encode(x_train)
        y_train_enc = encode(y_train)

        # Forward propagation
        a1_enc = x_train_enc.transpose()                                # (784,64)
        z2_enc = add_enc(TruncPrMul(np.dot(W2_enc, a1_enc)), b2_enc)    # (128,784) * (784,64)
        a2_enc = relu_enc(z2_enc)                                       # (128,64)
        z3_enc = add_enc(TruncPrMul(np.dot(W3_enc, a2_enc)), b3_enc)    # (10,128) * (128,64)
        a3_enc = relu_enc(z3_enc)                                       # (10,64)

        # Backward propagation
        delta_3_enc = sub_enc(a3_enc, y_train_enc.transpose())
        delta_2_enc = mul_enc(drelu_enc(z2_enc), TruncPrMul(np.dot(W3_enc.transpose(), delta_3_enc)))

        gradient_b3_enc = delta_3_enc
        gradient_b2_enc = delta_2_enc
        gradient_W3_enc = TruncPrMul(np.dot(delta_3_enc, a2_enc.transpose()))
        gradient_W2_enc = TruncPrMul(np.dot(delta_2_enc, a1_enc.transpose()))

        sum_gradient_b3_enc = np.sum(gradient_b3_enc, axis=1).reshape(-1, 1)
        sum_gradient_b2_enc = np.sum(gradient_b2_enc, axis=1).reshape(-1, 1)
        sum_gradient_W3_enc = gradient_W3_enc
        sum_gradient_W2_enc = gradient_W2_enc

        b3_enc = sub_enc(b3_enc, TruncPrMul(LR_enc * div_enc(sum_gradient_b3_enc, BATCH_SIZE)))
        b2_enc = sub_enc(b2_enc, TruncPrMul(LR_enc * div_enc(sum_gradient_b2_enc, BATCH_SIZE)))
        W3_enc = sub_enc(W3_enc, TruncPrMul(LR_enc * div_enc(sum_gradient_W3_enc, BATCH_SIZE)))
        W2_enc = sub_enc(W2_enc, TruncPrMul(LR_enc * div_enc(sum_gradient_W2_enc, BATCH_SIZE)))

        # Training Error
        sum_of_training_error += np.sum(np.argmax(decode(a3_enc), axis=0) != np.argmax(y_train.transpose(), axis=0)).astype(
            int)
        # print("W2_enc_update =", decode(W2_enc[0, 250:300]))

    test_layer_2_enc = relu_enc(add_enc(TruncPrMul(np.dot(W2_enc, encode(x_test_total).transpose())), b2_enc))
    test_layer_3_enc = relu_enc(add_enc(TruncPrMul(np.dot(W3_enc, test_layer_2_enc)), b3_enc))
    sum_of_test_error = np.sum(np.argmax(decode(test_layer_3_enc), axis=0) == np.argmax(y_test_total.transpose(), axis=0)).astype(int)
    print("epoch:", epoch, "training_error:", sum_of_training_error, "accuracy_rate:", sum_of_test_error / TEST_SIZE)
