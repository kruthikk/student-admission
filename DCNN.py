"""
https://hackernoon.com/learning-ai-if-you-suck-at-math-p5-deep-learning-and-convolutional-neural-nets-in-plain-english-cda79679bbe3
"""
from __future__ import print_function
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

np.random.seed(1337)
#layer 1 - edge +blob, layer2 = textture, object parts, object classes
#each layer responds/considers very specific feature/s

#define how many images we will process at once
batch_size=128

#define how many type of objects we can detect in this set
#since cIFAR 10 only detects 10 kinds of bjects, we set this t0 10
nb_classes = 10
#The epoch - define how long we train the system. Longer is not always bette as after a period of time, we reach the
#point of diminishing returns. Adjust this accordingly
nb_epoch = 45

#image dimensions. The image deimensions are 32*32
img_rows, img_cols = 32,32

#no of convolutional filters/layers to use
nb_filters = 32

#size of pooling area for max pooling
pool_size = (2,2)

#convolution kernel size. The smallest kernel size is 1*1 - which means that key features are only 1 pixel wide.
#Typical kernel size check for useful features over 3 pxels at a time and then pool those features down to 2*2 grid - pool size
kernel_size = (3,3)

#before adding layers load and process the data. cifar10 is about 186MB expanded

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

if K.image_dim_ordering() == 'th':
	X_train = X_train.reshape(X_train.shape[0],3,img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0],3,img_rows, img_cols)
	input_shape = (3, img_rows, img_cols)
else:
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,3)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)
	input_shape = (img_rows, img_cols, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape: ', X_train.shape)
print('X_train samples: ', X_train.shape[0])
print('X_test samples: ', X_test.shape[0])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Conv2D(nb_filters, kernel_size, strides=(1,1), padding = 'valid', input_shape= input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size, strides=(1,1), input_shape= input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(0.25))

model.add(Conv2D(nb_filters, kernel_size, strides=(1,1), padding = 'valid', input_shape= input_shape))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, kernel_size,strides=(1,1), input_shape= input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#tb = TensorBoard(log_dir = './logs')


model.fit(X_train, Y_train, batch_size= batch_size, nb_epoch = nb_epoch, verbose=1, validaton_data= (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print( 'Test score: ', score[0])
#print('Accuracy: ', score[-1])

print('dcnn demo done')