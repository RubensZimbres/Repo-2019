#! pip install --upgrade tfp-nightly
#! pip install tf_nightly
#! pip install tf_estimator_nightly

import tensorflow as tf
import tensorboard
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow import keras
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(42)
dataframe = pd.read_csv('Apple_Data_300.csv').ix[0:480,:]
dataframe.head()

plt.plot(range(0,dataframe.shape[0]),dataframe.iloc[:,1])

x1=np.array(dataframe.iloc[:,1]+np.random.randn(dataframe.shape[0])).astype(np.float32).reshape(-1,1)[0:350]

y=np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[0:350]

tfd = tfp.distributions

init = tf.global_variables_initializer()

epochs=1000

with tf.Session() as sess:
    sess.run(init)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(1,kernel_initializer='glorot_uniform'),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=negloglik)

    model.fit(x1,y, epochs=epochs, verbose=True, batch_size=16)
    
    yhat = model(np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[351:450])
    mean0 = tf.convert_to_tensor(yhat)
    mean = sess.run(mean0)    
    stddev = yhat.stddev()
    mean_plus_2_std = sess.run(mean - 3. * stddev)
    mean_minus_2_std = sess.run(mean + 3. * stddev)

from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.python import keras
import numpy as np

#from tensorflow_model_optimization.python.core.sparsity.keras import prune
#from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
#from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
#ConstantSparsity = pruning_schedule.ConstantSparsity

epochs = 1000
num_train_samples = x1.shape[0]
end_step = 1000
print('End step: ' + str(end_step))

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.10,
                                                   begin_step=700,
                                                   end_step=end_step,
                                                   frequency=100)
}

#pruning_params = {
#      'pruning_schedule': ConstantSparsity(0.75, begin_step=20, frequency=100)
#  }

tfd = tfp.distributions

input_shape=x1.shape

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    pruned_model = tf.keras.Sequential([
        sparsity.prune_low_magnitude(
            tf.keras.layers.Dense(10, activation='relu',kernel_initializer='glorot_uniform'),
            **pruning_params),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])
    
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
            
    callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='D:\Python\logs2', profile_batch=0)]

    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=negloglik)

    pruned_model.fit(x1,y, epochs=epochs, verbose=True, batch_size=16,callbacks=callbacks)
    
    yhat2 = pruned_model(np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[351:450])
    mean02 = tf.convert_to_tensor(yhat2)
    mean2 = sess.run(mean02)    
    stddev2 = yhat2.stddev()
    mean_plus_2_std2 = sess.run(mean2 - 3. * stddev2)
    mean_minus_2_std2 = sess.run(mean2 + 3. * stddev2)

plt.figure(figsize=(12,9))
plt.plot(y,color='red',linewidth=1,label='Real Data')
plt.plot(np.concatenate([y,mean_minus_2_std],axis=0),color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(np.concatenate([y,mean_plus_2_std],axis=0),color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.plot(np.concatenate([y,mean_minus_2_std],axis=0),color='red',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(np.concatenate([y,mean_plus_2_std],axis=0),color='red',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(mean,color='black',linewidth=3,label='Predicted + 2 std dev')
plt.plot(mean_minus_2_std,color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(mean_plus_2_std,color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.plot(mean_minus_2_std2,color='red',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(mean_plus_2_std2,color='red',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()
