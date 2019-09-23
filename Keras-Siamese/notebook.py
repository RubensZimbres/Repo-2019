from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers.core import Reshape
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D,Convolution1D,MaxPooling1D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import os
from keras.optimizers import SGD

with session.as_default():
    with session.graph.as_default():
        
        epochs=2500
        learning_rate = 0.01
        decay_rate = 5e-5
        momentum = 0.6

        sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)

        input_shape2=(20,20,1)
        input_img = Input(batch_shape=(None, 20,20,1))
        input_img1 = Input(batch_shape=(None, 20,20,1))
        input_img2 = Input(batch_shape=(None, 4,1))


        denoise_left=Convolution2D(20, 3,3,
                                border_mode='same',
                                input_shape=input_shape)(input_img)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('relu')(denoise_left)
        denoise_left=MaxPooling2D(pool_size=(2,2))(denoise_left)
        denoise_left=Convolution2D(20, 3, 3,
                                    init='glorot_uniform',border_mode='same')(denoise_left)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('relu')(denoise_left)
        denoise_left=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('relu')(denoise_left)
        denoise_left=UpSampling2D(size=(2, 2))(denoise_left)
        denoise_left=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('relu')(denoise_left)
        denoise_left=UpSampling2D(size=(2, 2))(denoise_left)
        denoise_left=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('relu')(denoise_left)
        denoise_left=Convolution2D(1, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left)
        denoise_left=BatchNormalization()(denoise_left)
        denoise_left=Activation('sigmoid')(denoise_left)
        denoise_left=Dense(20)(denoise_left)
        denoise_left=Reshape((-1,20,1))(denoise_left)
        denoise_left=Flatten()(denoise_left)

        denoise_left1=Convolution2D(20, 3,3,
                                border_mode='same',
                                input_shape=input_shape)(input_img1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('relu')(denoise_left1)
        denoise_left1=MaxPooling2D(pool_size=(2,2))(denoise_left1)
        denoise_left1=Convolution2D(20, 3, 3,
                                    init='glorot_uniform',border_mode='same')(denoise_left1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('relu')(denoise_left1)
        denoise_left1=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('relu')(denoise_left1)
        denoise_left1=UpSampling2D(size=(2, 2))(denoise_left1)
        denoise_left1=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('relu')(denoise_left1)
        denoise_left1=UpSampling2D(size=(2, 2))(denoise_left1)
        denoise_left1=Convolution2D(8, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('relu')(denoise_left1)
        denoise_left1=Convolution2D(1, 3, 3,init='glorot_uniform',border_mode='same')(denoise_left1)
        denoise_left1=BatchNormalization()(denoise_left1)
        denoise_left1=Activation('sigmoid')(denoise_left1)
        denoise_left1=Dense(20)(denoise_left1)
        denoise_left1=Reshape((-1,20,1))(denoise_left1)
        denoise_left1=Flatten()(denoise_left1)
        
        denoise_concat=Dense(20)(input_img2)
        denoise_concat=Activation('relu')(denoise_concat)
        denoise_concat=Dense(20)(denoise_concat)
        denoise_concat=Activation('relu')(denoise_concat)
        denoise_concat=Dense(4)(denoise_concat)
        denoise_concat=Reshape((-1,4,1))(denoise_concat)
        denoise_concat=Flatten()(denoise_concat)

        squeeze0=Concatenate()([denoise_left,denoise_left1,denoise_concat])
        squeeze0=Dropout(0.2)(squeeze0)
        squeeze0=Dense(20)(squeeze0)
        squeeze0=Activation('sigmoid')(squeeze0)
        squeeze0=Dense(2)(squeeze0)
        squeeze0=Activation('sigmoid')(squeeze0)

        model = Model(inputs = [input_img,input_img1,input_img2], outputs = squeeze0)

        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics = ['accuracy'])

        model.summary()        
        
from keras.utils import plot_model 
plot_model(model, to_file='model_gif.png')

with tf.Session(graph=graph, config=config) as session:
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init)
    model.fit([X_train, X_train1],y_train,
                nb_epoch=epochs,
                batch_size=2771,verbose=1,validation_split=0.1)
    preds=model.predict([X_test,X_test1])

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,f1_score,recall_score

confusion_matrix(y_test.T[1],np.argmax(preds,axis=1))
pred0=np.argmax(preds,axis=1)
Y=y_test.T[1]
print('\n')
print('Accuracy:',accuracy_score(np.array(Y),pred0))
print('Precision (FP):',precision_score(np.array(Y),pred0,average='binary'))
print('Recall (FN):',recall_score(np.array(Y),pred0,average='binary'))
print('f1:',f1_score(np.array(Y),pred0,average='binary'))
print(confusion_matrix(np.array(Y),pred0),'\n')
