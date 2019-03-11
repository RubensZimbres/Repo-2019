import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import GlobalAveragePooling2D
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

x_train_CNN=x_train.reshape(60000,28,28,1)

y_train2=pd.get_dummies(y_train)

epochs=3
learning_rate = 0.07
decay_rate = 5e-5
momentum = 0.6

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

input_shape=(28,28,1)

input_img = Input(batch_shape=(None, 28,28,1))
squeeze=Lambda(lambda x: x ** 2,input_shape=(784,),output_shape=(1,784))(input_img)
squeeze=Reshape((28,28,1))(squeeze)
squeeze=Conv2D(64, 3,3,
                          border_mode='valid',
                        input_shape=input_shape)(squeeze)
squeeze=BatchNormalization()(squeeze)
squeeze=ELU(alpha=1.0)(squeeze)
squeeze=MaxPooling2D(pool_size=(2,2))(squeeze)
squeeze=Conv2D(32, 1, 1,
                            init='glorot_uniform')(squeeze)
squeeze=BatchNormalization()(squeeze)
squeeze=ELU(alpha=1.0)(squeeze)

squeeze_left=squeeze
squeeze_left=Conv2D(64, 3,3,
                          border_mode='valid',
                        input_shape=input_shape)(squeeze_left)
squeeze_left=ELU(alpha=1.0)(squeeze_left)

squeeze_right=squeeze
squeeze_right=Conv2D(64, 3,3,
                          border_mode='valid',
                        input_shape=input_shape)(squeeze_right)
squeeze_right=ELU(alpha=1.0)(squeeze_right)

squeeze0=Concatenate()([squeeze_left,squeeze_right])
squeeze0=Dropout(0.2)(squeeze0)
squeeze0=GlobalAveragePooling2D()(squeeze0)
squeeze0=Dense(10)(squeeze0)
squeeze0=Activation('sigmoid')(squeeze0)

model = Model(inputs = input_img, outputs = squeeze0)

model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])
model.summary()

target=np.array(y_train2)
proportion=[]
for i in range(0,len(target)):
    proportion.append([i,len(np.where(target==np.unique(target)[0])[0])/target.shape[0]])

class_weight = dict(proportion)

model.fit(x_train_CNN,np.array(y_train2),
                nb_epoch=15,
                batch_size=30,verbose=1,class_weights=class_weight)

predictions=np.argmax(model.predict(x_train_CNN,verbose=1),axis=1)
