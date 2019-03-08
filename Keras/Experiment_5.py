import pandas as pd  
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

data0=pd.read_csv("qui5.csv",sep=',')

data=data0.iloc[::-1]

data = data.astype('int')

dados=data.iloc[:,1:]

dados.values.sort(axis = 1)

timesteps = 1
look_back=timesteps

trainX=[np.array(dados)[x:x+timesteps] for x in range(0,dados.shape[0]-timesteps-1)]

trainX=np.array(trainX).reshape(-1,timesteps,5)

trainY=np.array(dados.iloc[look_back+1:,:])#.reshape(-1,1,5)

X0=trainX
Y0=trainY

from keras.models import Sequential  
from keras.layers.core import Dense, Activation,Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD

learning_rate = 0.008
decay_rate = 5e-6
momentum = 0.65

data_dim = 5
batchsize=X0.shape[0]

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

import math
nodes=int((math.factorial(80)/math.factorial(80-5))**(1/3))

input_img = Input(batch_shape=(None, timesteps,5))
model1=LSTM(nodes, return_sequences=True,input_shape=(timesteps, data_dim), activation='relu')(input_img)
model2=LSTM(nodes,return_sequences=True, activation='relu')(model1)
model3=LSTM(nodes,return_sequences=False, activation='relu')(model2)
model4=Dense(5, activation='relu')(model3)
model = Model(inputs = input_img, outputs = model4)

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)


model.compile(loss="mean_squared_error", optimizer="rmsprop",metrics=['mae'])  

model.summary()

epochs_=30


model.fit(X0.reshape(-1,timesteps,5).astype("float"),Y0.astype("float"), batch_size=batchsize, epochs=epochs_,verbose=1)  

testX=[np.array(dados)[x:x+timesteps] for x in range(0,dados.shape[0]-timesteps)]

model.predict(np.array(testX).reshape(-1,timesteps,5).astype("float"))

