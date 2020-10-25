from numpy import *
random.seed(1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers
from keras.utils import np_utils
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data() # X_train.shape (60000, 28, 28)
# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28 * 28) # flatten X
X_test = X_test.reshape(X_test.shape[0], 28 * 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)  # shape (60000, 10)
Y_test = np_utils.to_categorical(y_test, 10)
# Define model architecture
model = Sequential()
model.add(Dense(300, activation='tanh', input_shape=(28 * 28,)))
model.add(Dense(10, activation="softmax" ))

model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'] )
# fit and train
hist = model.fit(X_train, Y_train, epochs = 10, batch_size=50)

hist.history
# evaluate
test_loss, test_acc = model.evaluate(X_test, Y_test)
test_acc












