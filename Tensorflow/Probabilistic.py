import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
np.random.seed(42)
dataframe = pd.read_csv('Apple_Data_300.csv').ix[0:800,:]
dataframe.head()

plt.plot(range(0,dataframe.shape[0]),dataframe.iloc[:,1])

x1=np.array(dataframe.iloc[:,1]+np.random.randn(dataframe.shape[0])).astype(np.float32).reshape(-1,1)

y=np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)

tfd = tfp.distributions

model = tf.keras.Sequential([
  tf.keras.layers.Dense(1,kernel_initializer='glorot_uniform'),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
])

negloglik = lambda x, rv_x: -rv_x.log_prob(x)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=negloglik)

model.fit(x1,y, epochs=300, verbose=True)

yhat = model(x1)
mean = yhat.mean()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    mm = sess.run(mean)    
    mean = yhat.mean()
    stddev = yhat.stddev()
    mean_plus_2_std = sess.run(mean - 2. * stddev)
    mean_minus_2_std = sess.run(mean + 2. * stddev)


plt.figure(figsize=(13,10))
plt.plot(y,color='red')
#plt.plot(mm)
plt.plot(mean_minus_2_std,color='blue',linewidth=1)
plt.plot(mean_plus_2_std,color='blue',linewidth=1)
