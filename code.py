from pylab import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# Load the MNIST digit data
M = loadmat("mnist_all.mat")

######################
#                    P A R T  1                     #
######################

# Data preprocessing
for i in range(0,10):
     M["train" + str(i)] = np.around(M["train" + str(i)] / 255.0)
     M["test" + str(i)] = np.around(M["test" + str(i)] / 255.0)

training_data = M["train0"]
for i in range(1, 10):
    training_data = np.concatenate((training_data, M["train" + str(i)]), axis=0)
    # concatenate 连结 axis = 0 means combine by rows
len(training_data)

test_data = M["test0"]
for i in range(1, 10):
    test_data = np.concatenate((test_data, M["test" + str(i)]), axis=0)
len(test_data)

training_label = np.zeros([len(M["train0"]), 1])
for i in range(1, 10):
    training_label = np.concatenate((training_label, i * np.ones([len(M["train" + str(i)]), 1])), axis=0)
label_matrix = zeros((10, 60000))
for i in range(10):
    for j in range(60000):
        if training_label[j] == i:
            label_matrix[i, j] = 1

test_label = np.zeros([len(M["test0"]), 1])
for i in range(1, 10):
    test_label = np.concatenate((test_label, i * np.ones([len(M["test" + str(i)]), 1])), axis=0)
testlabel_matrix = zeros((10, 10000))
for i in range(10):
    for j in range(10000):
        if test_label[j] == i:
            testlabel_matrix[i, j] = 1

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

# defines some functions we need
def softmax(x):  # note: we need to replace x here with w@x.T+b in practice
    return exp(x) / tile(sum(exp(x), 0), (len(x), 1))

def label(x): pass

def loss(y, y_):
    # y_ is the probability matrix of w@x.T+b, ie. softmax(w@x.T+b)
    # y is the true label matrix of x, which is composed by 0, 1.
    return  - sum(y * log(y_))

######################
#                    P A R T  3                     #
######################
def gradient_w(y_, y, x):
    return (y_ - y) @ x

def gradient_b(y_, y):
    return sum((y_ - y), 1).reshape((10,1))

def accuracy(y, y_):
    y_index = argmax(y_, axis = 0)
    s = 0
    for i in range(len(y_index)):
        if y[y_index[i], i] == 1:
            s = s + 1
    return s / len(y_index)

######################
#                    P A R T  4                     #
######################

# generate a subset with 100 samples
random.seed(1)
index_100 = random.randint(60000, size=100)
train100 = training_data[index_100, :]
y100 = label_matrix[:, index_100]
# initialize the w b
w = ones((10, 784))
b = ones((10, 1))
y100_ = softmax(w @ train100.T + b)
loss_100 = loss(y100, y100_)
loss_100 # loss is 230
# calculate the gradient
deriv_w_100 = gradient_w(y100_, y100, train100)
deriv_b_100 = gradient_b(y100_, y100)
w1 = w - 0.01 * deriv_w_100
b1 = b - 0.01 * deriv_b_100
loss(y100, softmax(w1 @ train100.T + b1)) # loss is 142, which is less than before
# If we don't use the functions we defined
where(deriv_w_100 == amax(deriv_w_100)) # w[0,434] changed a lot (the largest perhaps)
w = ones((10, 784))
b = ones((10, 1))
w[0,434] = 1.01
(loss(y100, softmax(w @ train100.T + b)) - loss_100) / 0.01 # 6.0270 by differencing
deriv_w_100[0, 434] # 5.9999
w = ones((10, 784))
b = ones((10, 1))
w[5,555] = 1.01
(loss(y100, softmax(w @ train100.T + b)) - loss_100) / 0.01 # 0.10045
deriv_w_100[5, 555] # 0.09999
w = ones((10, 784))
b = ones((10,1))
b[1] = 1.01
(loss(y100, softmax(w @ train100.T + b)) - loss_100) / 0.01 # -3.9548 by differencing
deriv_b_100[1] # -4.0
w = ones((10, 784))
b = ones((10,1))
b[5] = 1.01
(loss(y100, softmax(w @ train100.T + b)) - loss_100) / 0.01 # -4.9548 by differencing
deriv_b_100[5] # -5.0

######################
#                    P A R T  5                     #
######################

# shuffle our data and construct mini-batches
def random_mini_batch(x, y, times, size = 50):
    random.seed(1)
    mini_batches = []
    for i in range(times):
        index = random.randint(60000, size=size)
        mini_batch_x = x[index, :]
        mini_batch_y = y[:, index]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches

w = ones((10, 784))
b = ones((10,1))
# update by mini-batch GD
seed = 0
parameter = [(w,b)]
minibatches = random_mini_batch(training_data, label_matrix, times = 6000, size=50) # update 1000 times
for j in range(len(minibatches)):
    (minibatch_x, minibatch_y) = minibatches[j]
    deriv_w = gradient_w(softmax(w @ minibatch_x.T + b), minibatch_y, minibatch_x)
    deriv_b = gradient_b(softmax(w @ minibatch_x.T + b), minibatch_y)
    w = w - 0.01 * deriv_w
    b = b - 0.01 * deriv_b
    parameter.append((w, b))

# loss curve
loss_train = []
loss_test = []
for i in range(len(parameter)):
    (w, b) = parameter[i]
    L_train = loss(label_matrix, softmax(w @ training_data.T + b))
    L_test = loss(testlabel_matrix, softmax(w @ test_data.T + b))
    loss_train.append(L_train)
    loss_test.append(L_test)

plt.plot(range(len(parameter)), loss_train)
plt.tick_params(labelsize=30)
plt.title("training data", size=30)

plt.plot(range(len(parameter)), loss_test, color= "red")
plt.tick_params(labelsize=30)
plt.title("test data", size=30)


plt.plot(range(len(parameter)), loss_train, label="train")
plt.plot(range(len(parameter)), loss_test, color= "red", label="test")
plt.legend(prop={"size":20})
plt.show()

# correct rate
accuracy_train = []
accuracy_test = []
for i in range(len(parameter)):
    (w, b) = parameter[i]
    acc_train = accuracy(label_matrix, softmax(w @ training_data.T + b))
    acc_test = accuracy(testlabel_matrix, softmax(w @ test_data.T + b))
    accuracy_train.append(acc_train)
    accuracy_test.append(acc_test)

plt.title("training data", size=30)
plt.tick_params(labelsize=30)
plt.plot(range(len(parameter)), accuracy_train)

plt.title("test data", size=30)
plt.tick_params(labelsize=30)
plt.plot(range(len(parameter)), accuracy_test, color="red")


plt.plot(range(len(parameter)), accuracy_train, label="train")
plt.plot(range(len(parameter)), accuracy_test, color="red", lw=0.5, label="test")
plt.legend(prop = {"size" :16})
plt.show()

# show some examples for correct classifications and the incorrect
cor_test = []
cor_index = []
incor_test = []
incor_index = []
incor_true_index = []
w = parameter[-1][0]
b = parameter[-1][1]
prob = softmax(w @ test_data.T + b)
for i in range(len(test_data)):
    if argmax(prob, 0)[i] == test_label[i]:
        cor_test.append(test_data[i])
        cor_index.append(test_label[i])
    else:
        incor_test.append(test_data[i])
        incor_index.append(argmax(prob, 0)[i])
        incor_true_index.append(test_label[i])

cor_sub = []
cor_sub_index = []
random.seed(111)
r1 = randint(len(cor_test), size=20)
for i in range(20):
    cor_sub.append(cor_test[r1[i]])
    cor_sub_index.append(cor_index[r1[i]])

incor_sub = []
incor_sub_index = []
incor_sub_true_index = []
random.seed(222)
r2 = randint(len(incor_test), size=10)
for i in range(10):
    incor_sub.append(incor_test[r2[i]])
    incor_sub_index.append(incor_index[r2[i]])
    incor_sub_true_index.append(int(incor_true_index[r2[i]]))

f1, fig1 = plt.subplots(2,10, figsize=(40,4.4))
for i in range(2):
    for j in range(10):
        fig1[i, j].imshow(cor_sub[i * 10 + j].reshape((28, 28)), cmap = cm.gray)
        fig1[i, j].set_title(str(int(cor_sub_index[i * 10 + j])), size=30)
        fig1[i, j].set_yticklabels([])
        fig1[i, j].set_xticklabels([])

f2, fig2 = plt.subplots(1, 10, figsize=(20, 2.2))
for i in range(10):
    fig2[i].imshow(incor_sub[i].reshape((28, 28)), cmap = cm.gray)
    fig2[i].set_title("P"+str(incor_sub_index[i])+"  T"+str(incor_sub_true_index[i]), size=30)
    fig2[i].set_yticklabels([])
    fig2[i].set_xticklabels([])

######################
#                    P A R T  6                     #
######################

# Visualization
f, fig = plt.subplots(2, 5, sharey=True, figsize=(10, 4) )
for i in range(2):
    for j in range(5):
        fig[i,j].imshow(parameter[-1][0][i*5+j, :].reshape((28, 28)), cmap=cm.coolwarm)
        fig[i,j].set_title(str(i*5+j))



from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

######################
#                    P A R T  7                     #
######################

network = models.Sequential()
network.add(layers.Dense(300, activation='tanh', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
opt = SGD(lr=0.01)
network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

######################
#                    P A R T  8                     #
######################

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels_one = np_utils.to_categorical(train_labels, 10)
test_labels_one = np_utils.to_categorical(test_labels, 10)
history = network.fit(train_images, train_labels_one, epochs=15, batch_size=50,
                      validation_data=(test_images, test_labels_one), shuffle=True)

# plot accuracy curve
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('part8learning_curve' + ".png", bbox_inches='tight')

# plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('part8cost_curve' + ".png", bbox_inches='tight')
plt.tight_layout()

# plot failure and success fig
mnist_model = network
predicted_classes = mnist_model.predict_classes(test_images)
correct_indices = np.nonzero(predicted_classes == test_labels)[0]
incorrect_indices = np.nonzero(predicted_classes != test_labels)[0]

print(len(correct_indices), " classified correctly")
print(len(incorrect_indices), " classified incorrectly")

fig.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.5)
tt2 = plt.suptitle('Success Cases')
tt2.set_position([.45, .6])
for i, correct in enumerate(correct_indices[:20]):
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_images[correct].reshape(28, 28), cmap=cm.gray)
    plt.title("{}".format(predicted_classes[correct]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('part8success' + ".png", bbox_inches='tight')

fig.tight_layout()
ttl = plt.suptitle('Failure Cases')
ttl.set_position([.5, .7])
for i, incorrect in enumerate(incorrect_indices[:10]):
    plt.subplot(1, 10, i + 1)
    plt.imshow(test_images[incorrect].reshape(28, 28), cmap=cm.gray)
    plt.title("P{} T{}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('part8failure' + ".png", bbox_inches='tight')

######################
#                    P A R T  9                     #
######################

weights, bias = network.layers[0].get_weights()
wei, bia = network.layers[1].get_weights()
w = weights.reshape((28, 28, 300))
w_out = wei
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# select most influential neuron for each acto:
name = digits[5]

neuron1 = np.argmax(w_out.T[5])
im_1 = w[:, :, neuron1] + bias[neuron1]
p1 = plt.imshow(im_1, cmap=plt.cm.coolwarm)
plt.title('maximum influence weights of digit 5')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.savefig('P9weights1' + ".png", bbox_inches='tight')
plt.show()
print(w_out[neuron1])

neuron2 = argmin(w_out.T[5])
im_2 = w[:, :, neuron2] + bias[neuron2]
p2 = plt.imshow(im_2, cmap=plt.cm.coolwarm)
plt.title('minimum influence weights of digit 5')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.savefig('P9weights2' + ".png", bbox_inches='tight')
plt.show()
print(w_out[neuron2])