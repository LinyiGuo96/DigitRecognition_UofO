import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.io import loadmat
import random

# Load the MNIST digit data
M = loadmat("mnist_all.mat")
# Data preprocessing
for i in range(0,10):
     M["train" + str(i)] = np.around(M["train" + str(i)] / 255.0)

training_data = M["train0"]
for i in range(1, 10):
    training_data = np.concatenate((training_data, M["train" + str(i)]), axis=0)
    # concatenate 连结 axis = 0 means combine by rows

training_label = np.zeros([len(M["train0"]), 1])
for i in range(1,10):
    training_label = np.concatenate((training_label, i * np.ones([len(M["train"+str(i)]),1])), axis = 0)

######################
#                    P A R T  1                     #
######################

# display a 10 * 10 matrix of digits
f, pic = plt.subplots(10,10)
np.random.seed(1)
for i in range(0,10): # i means the digit
     for j in range(0,10): # j means the j^th pic of digit i
         pic[i,j].imshow(M["train" + str(i)][random.randint(1, len(M["train" + str(i)]))].reshape((28,28)), cmap = cm.gray)
         pic[i, j].set_yticklabels([])  # hide the axis values
         pic[i, j].set_xticklabels([])  # tick 记号

######################
#                    P A R T  2                     #
######################

def softmax(x):
    return exp(x) / tile(sum(exp(x),0), (len(x),1))

# the loss function of the whole training data
I0 = zeros((10,60000))
for i in range(10):
    for j in range(60000):
        if training_label[j] == i:
            I0[i, j] = 1

def loss0(w, b):
    L0 = - sum( I0 * log(softmax(w @ training_data.T + b)))
    return L0

######################
#                    P A R T  3                     #
######################

# note: numpy doesn't differentiate row/col vector,
# py will auto-recognize the dim of a vector based on its position.
# and py will auto-fill the blank of a vector or a matrix if need
def gradient_w(w, b):
    s1 = zeros((1, 784))
    s2 = zeros((1, 784))
    for i in range(0, 10):
        for j in range(0, 60000):
            s2 = s2 +  (softmax((w @ training_data[j, :] + b.T).T)[i]) * training_data[j, :]
        s2 = s2 - sum(M["train" + str(i)], 0)
        s1 = np.concatenate((s1, s2), axis = 0)
    return s1[1:11, :]

def gradient_b(w, b):
    s1 = zeros((10,1))
    for i in range(0,10):
        s1[i] = sum(softmax(w @ training_data.T + b)[i, :]) - len(M["train" + str(i)])
    return s1
w = ones((10, 784))
b = ones((10, 1))
w_d = gradient_w(w, b)
b_d = gradient_b(w, b)
l0 = loss0(w, b)
w_d.shape
w1 = w - 0.01 * w_d
b1 = b - 0.01 * b_d
w1.shape
b1.shape
l1 = loss0(w1,b1)


######################
#                    P A R T  4                     #
######################
w = ones((10, 784))
b = ones((10, 1))
a = gradient_w(w, b)
l = loss0(w, b)
w_deriv = []
for i in range(0,10):
    w_deriv.append(a[i, i + 392])
w_deriv
w_appro = []
for i in range(0,10):
    w = ones((10, 784))
    w[i, i + 392] = 1.01
    w_appro.append((loss0(w,b) - l) / 0.01)
w_appro
asarray(w_appro) - asarray(w_deriv)

# b
w = ones((10, 784))
b = ones((10, 60000))
b0 = gradient_b(w, b)[:, 0]

b1 = []
for i in range(10):
    b = ones((10, 60000))
    b[i, :] = 1.01 * ones((1, 60000))
    b1.append((loss0(w, b) - l) / 0.01 )
asarray(b1) - b0
b0
b1








