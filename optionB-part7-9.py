from pylab import *
from numpy import *
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.optimizers import SGD
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

######################
#                    P A R T  7                     #
######################

network = models.Sequential()
network.add(layers.Dense(300, activation='tanh', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
opt=SGD(lr=0.01)
network.compile(optimizer=opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

######################
#                    P A R T  8                     #
######################

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
train_labels_one = np_utils.to_categorical(train_labels, 10)
test_labels_one = np_utils.to_categorical(test_labels, 10)
history=network.fit(train_images, train_labels_one, epochs=15, batch_size=50,validation_data=(test_images,test_labels_one),shuffle=True)

# plot accuracy curve
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('part8learning_curve'+".png", bbox_inches='tight')

# plot loss curve
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('part8cost_curve'+".png", bbox_inches='tight')
plt.tight_layout()

# plot failure and success fig
mnist_model = network
predicted_classes = mnist_model.predict_classes(test_images)
correct_indices = np.nonzero(predicted_classes == test_labels)[0]
incorrect_indices = np.nonzero(predicted_classes != test_labels)[0]

print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

fig.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.5)
tt2=plt.suptitle('Success Cases')
tt2.set_position([.45, .6])
for i, correct in enumerate(correct_indices[:20]):
    plt.subplot(2,10,i+1)
    plt.imshow(test_images[correct].reshape(28,28), cmap = cm.gray)
    plt.title("{}".format(predicted_classes[correct]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('part8success'+".png", bbox_inches='tight')


fig.tight_layout()
ttl=plt.suptitle('Failure Cases')
ttl.set_position([.5, .7])
for i, incorrect in enumerate(incorrect_indices[:10]):
    plt.subplot(1,10,i+1)
    plt.imshow(test_images[incorrect].reshape(28,28),cmap = cm.gray)
    plt.title("P{} T{}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('part8failure'+".png", bbox_inches='tight')

######################
#                    P A R T  9                     #
######################

weights, bias = network.layers[0].get_weights()
wei,bia=network.layers[1].get_weights()
w = weights.reshape((28, 28, 300))
w_out =wei
digits = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']

    
# select most influential neuron for each acto:
name = digits[5]

neuron1 = np.argmax(w_out.T[5])
im_1 = w[:,:,neuron1] + bias[neuron1]
p1 =plt.imshow(im_1,cmap = plt.cm.coolwarm)
plt.title('maximum influence weights of digit 5')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.savefig('P9weights1'+".png", bbox_inches='tight')
plt.show()
print(w_out[neuron1])

neuron2 = argmin(w_out.T[5])
im_2 = w[:,:,neuron2] + bias[neuron2]
p2 =plt.imshow(im_2,cmap = plt.cm.coolwarm)
plt.title('minimum influence weights of digit 5')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.savefig('P9weights2'+".png", bbox_inches='tight')
plt.show()
print(w_out[neuron2])