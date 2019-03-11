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

learning_rate = 0.01
decay_rate = 5e-6
momentum = 0.65

data_dim = 5
timesteps = 1
num_classes = 10
batchsize=200

input_img = Input(batch_shape=(None, 1,5))
model1=LSTM(128, return_sequences=True,input_shape=(timesteps, data_dim))(input_img)
model2=LSTM(128,return_sequences=False)(model1)
model4=Dense(5, activation='relu')(model2)
model = Model(inputs = input_img, outputs = model4)

from keras.utils import plot_model
import pydot
import graphviz
import pydot_ng as pydot

plot_model(model,to_file="model.png")

sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)


model.compile(loss="mean_squared_error", optimizer="rmsprop",metrics=['mae'])  

model.summary()

epochs_=60

model.fit(X0.reshape(-1,1,5).astype("float"),Y0.astype("float"), batch_size=batchsize, epochs=epochs_,verbose=1)  

model.predict(np.array(dados).reshape(-1,1,5).astype("float"))
