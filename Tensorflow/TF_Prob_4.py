%reset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.random.seed(42)
dataframe = pd.read_csv('Apple_Data_300.csv').ix[0:800,:]
dataframe.head()

plt.plot(range(0,dataframe.shape[0]),dataframe.iloc[:,1])


x1=np.array(dataframe.iloc[:,[1,2,3,5]]).astype(np.float32).reshape(-1,4)[0:400]

y=np.array(dataframe.iloc[:,4]).T.astype(np.float32).reshape(-1,1)[5:405]

y.shape

tfd = tfp.distributions

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(4,kernel_initializer='glorot_uniform'),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=negloglik)
    
    model.fit(x1,y, epochs=100, verbose=True, batch_size=16)
    
    yhat = model(np.array(dataframe.iloc[:,[1,2,3,5]]).T.astype(np.float32).reshape(-1,4)[406:500])
    mean0 = tf.convert_to_tensor(yhat)
    mean = sess.run(mean0)    
    stddev = yhat.stddev()
    mean_plus_2_std = sess.run(mean - 3. * stddev).mean(axis=1)
    mean_minus_2_std = sess.run(mean + 3. * stddev).mean(axis=1)

mean_minus_2_std

mean_minus_2_std.shape
yhat.shape

plt.figure(figsize=(12,9))
plt.plot(y,color='red',linewidth=1,label='Real Data')
#plt.plot(mm,color='green',linewidth=1,label='Real Data')
plt.plot(np.concatenate([y.reshape(1,-1)[0],mean_minus_2_std],axis=0),color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(np.concatenate([y.reshape(1,-1)[0],mean_plus_2_std],axis=0),color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
#plt.plot(mean,color='black',linewidth=3,label='Predicted + 2 std dev')
plt.plot(mean_minus_2_std,color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(mean_plus_2_std,color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()
