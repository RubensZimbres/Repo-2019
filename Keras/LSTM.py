import pandas as pd  
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

data0=pd.read_csv("qui5_all.csv",sep=',')

data=data0.iloc[::-1]

data = data.astype('int')

dados=data.iloc[:,1:]

dados.values.sort(axis = 1)

look_back=1

trainX=np.array(dados.iloc[0:-look_back])#.reshape(-1,1,5)

trainY=np.array(dados.iloc[look_back:])#.reshape(-1,1,5)

X0=trainX
Y0=trainY

from keras.models import Sequential  
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD

learning_rate = 0.015
decay_rate = 5e-6
momentum = 0.9

data_dim = 5
timesteps = 1
num_classes = 10
batchsize=300

input_img = Input(batch_shape=(None, 1,5))
model1=LSTM(12, return_sequences=False,input_shape=(timesteps, data_dim))(input_img)
model4=Dense(5, activation='relu')(model1)
model5 = Model(inputs = input_img, outputs = model4)

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)


model5.compile(loss="mean_squared_error", optimizer="rmsprop",metrics=['accuracy'])  

model5.summary()

epochs_=100

model5.fit(X0.reshape(-1,1,5).astype("float"),Y0.astype("float"), batch_size=batchsize, epochs=epochs_,verbose=1)  

model5.predict(np.array(dados).reshape(-1,1,5).astype("float"))


