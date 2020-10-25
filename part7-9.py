from numpy import *
random.seed(521)
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scipy.io import loadmat
import matplotlib.pyplot as plt
from keras.models import load_model

# import our data
M = loadmat("mnist_all.mat")

training_data = M["train0"]
for i in range(1, 10):
    training_data = concatenate((training_data, M["train" + str(i)]), axis=0)
    # concatenate 连结 axis = 0 means combine by rows

training_label = zeros([len(M["train0"]), 1])
for i in range(1, 10):
    training_label = concatenate((training_label, i * ones([len(M["train" + str(i)]), 1])), axis=0)

test_data = M["test0"]
for i in range(1, 10):
    test_data = concatenate((test_data, M["test" + str(i)]), axis=0)

test_label = zeros([len(M["test0"]), 1])
for i in range(1, 10):
    test_label = concatenate((test_label, i * ones([len(M["test" + str(i)]), 1])), axis=0)

# preprocess our data
training_data = training_data / 255.0
test_data = test_data / 255.0
training_label = to_categorical(training_label, 10)
test_label = to_categorical(test_label, 10)


# build and train our model
model = Sequential()
model.add(Dense(300, activation="tanh", input_shape=(28*28,)))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
m = model.fit(training_data, training_label, epochs=30, verbose=2, batch_size=50)

# test
model.evaluate(test_data, test_label)
plt.plot(range(len(m.history["loss"])), asarray(m.history["loss"]))
plt.plot(range(len(m.history["acc"])), asarray(m.history["acc"]))

# use train_on_batch to train our model
acc_train = []
acc_test = []
model = Sequential()
model.add(Dense(300, activation="tanh", input_shape=(28 * 28,)))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
random.seed(111)
for k in range(100):
    index = random.randint(60000, size=256)
    model.train_on_batch(training_data[index, :], training_label[index, :])
    output_train = model.evaluate(training_data, training_label, batch_size=10000,verbose=2)
    output_test = model.evaluate(test_data, test_label, batch_size=10000, verbose=2)
    acc_train.append(output_train[1])
    acc_test.append(output_test[1])
model.save("m_bz256_k1000.h5")

plt.subplot(211)
plt.plot(range(100), acc_train)
plt.title("train batch_size(256)")
plt.subplot(212)
plt.plot(range(100), acc_test, color="red")
plt.title("test batch_size(256)")

plt.plot(range(100), acc_train, label="train")
plt.plot(range(100), acc_test, color="red", label="test")
plt.legend()


