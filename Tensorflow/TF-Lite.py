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
import tempfile
import zipfile
import os

tf.disable_v2_behavior()

np.random.seed(42)
dataframe = pd.read_csv('Apple_Data_300.csv').ix[0:480,:]
dataframe.head()

plt.plot(range(0,dataframe.shape[0]),dataframe.iloc[:,1])

x1=np.array(dataframe.iloc[:,1]+np.random.randn(dataframe.shape[0])).astype(np.float32).reshape(-1,1)[0:350]

y=np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[0:350]

x_test=x1
y_test=y

tfd = tfp.distributions


epochs=20

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(5,kernel_initializer='glorot_uniform'),
      tf.keras.layers.Dense(5,kernel_initializer='glorot_uniform')
    ])
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mean_squared_error')

    model.fit(x1,y, epochs=epochs, verbose=True, batch_size=16)
    
    yhat = model(np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[351:450])
    mean0 = tf.convert_to_tensor(yhat)
    mean = sess.run(mean0)    
    stddev = np.std(sess.run(tf.convert_to_tensor(yhat)))
    mean_plus_2_std = mean - 3. * stddev
    mean_minus_2_std = mean - 3. * stddev
    _, keras_file = tempfile.mkstemp('.h5')
    print('Saving model to: ', keras_file)
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)

################################

from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.python import keras
import numpy as np

#from tensorflow_model_optimization.python.core.sparsity.keras import prune
#from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
ConstantSparsity = pruning_schedule.ConstantSparsity

num_train_samples = x1.shape[0]
end_step = 20
print('End step: ' + str(end_step))

pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.9,
                                                   begin_step=15,
                                                   end_step=end_step,
                                                   frequency=100)
}

pruning_params = {
      'pruning_schedule': ConstantSparsity(0.75, begin_step=20, frequency=100)
  }

tfd = tfp.distributions

input_shape=x1.shape

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    pruned_model = tf.keras.Sequential([
        sparsity.prune_low_magnitude(
            tf.keras.layers.Dense(20, activation='relu',kernel_initializer='glorot_uniform'),
            **pruning_params),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1))
    ])
    
    negloglik = lambda x, rv_x: -rv_x.log_prob(x)
            
    callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir='D:\Python\logs2', profile_batch=0)]

    pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=negloglik)

    pruned_model.fit(x1,y, epochs=epochs, verbose=True, batch_size=x1.shape[0],callbacks=callbacks)
    
    yhat2 = pruned_model(np.array(dataframe.iloc[:,1]).T.astype(np.float32).reshape(-1,1)[351:450])
    mean02 = tf.convert_to_tensor(yhat2)
    mean2 = sess.run(mean02)    
    stddev2 = np.std(sess.run(tf.convert_to_tensor(yhat2)))
    mean_plus_2_std2 = mean2 - 3. * stddev2
    mean_minus_2_std2 = mean2 + 3. * stddev2    

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    print('Saving pruned model to: ', pruned_keras_file)
    tf.keras.models.save_model(pruned_model, pruned_keras_file, include_optimizer=False)
    
    final_model = sparsity.strip_pruning(pruned_model)
    _, final_keras_file = tempfile.mkstemp('.h5')
    print('Saving pruned model to: ', pruned_keras_file)
    tf.keras.models.save_model(final_model, final_keras_file, include_optimizer=False)


################################
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    tf.keras.models.save_model(final_model, pruned_keras_file, include_optimizer=False)

    _, zip1 = tempfile.mkstemp('.zip') 
    with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(keras_file)
    print("Size of the unpruned model before compression: %.2f Mb" % 
          (os.path.getsize(keras_file) / float(2**20)))
    print("Size of the unpruned model after compression: %.2f Mb" % 
          (os.path.getsize(zip1) / float(2**20)))
    
    _, zip2 = tempfile.mkstemp('.zip') 
    with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(pruned_keras_file)
    print("Size of the pruned model before compression: %.2f Mb" % 
          (os.path.getsize(pruned_keras_file) / float(2**20)))
    print("Size of the pruned model after compression: %.2f Mb" % 
          (os.path.getsize(zip2) / float(2**20)))

###############################



'''STOP HERE'''

loaded_model = tf.keras.models.load_model(keras_file)


tflite_model_file = '/tmp/sparse_mnist.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
with open(tflite_model_file, 'wb') as f:
  f.write(tflite_model)
  
_, zip_tflite = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(tflite_model_file)
print("Size of the tflite model before compression: %.2f Mb" 
      % (os.path.getsize(tflite_model_file) / float(2**20)))
print("Size of the tflite model after compression: %.2f Mb" 
      % (os.path.getsize(zip_tflite) / float(2**20)))

import numpy as np

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def eval_model(interpreter, x_test, y_test):
  total_seen = 0
  num_correct = 0

  for img, label in zip(x_test, y_test):
    inp = img.reshape((1, 28, 28, 1))
    total_seen += 1
    interpreter.set_tensor(input_index, inp)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if np.argmax(predictions) == np.argmax(label):
      num_correct += 1

    if total_seen % 1000 == 0:
        print("Accuracy after %i images: %f" %
              (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)

print(eval_model(interpreter, x_test, y_test))
  
  
  
  
  

  
  
plt.figure(figsize=(12,9))
plt.plot(y,color='red',linewidth=1,label='Real Data')
plt.plot(np.concatenate([y,mean_minus_2_std],axis=0),color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(np.concatenate([y,mean_plus_2_std],axis=0),color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.plot(np.concatenate([y,mean_minus_2_std2[0].reshape(-1,1)],axis=0),color='black',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(np.concatenate([y,mean_plus_2_std2[0].reshape(-1,1)],axis=0),color='blue',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(mean,color='black',linewidth=3,label='Predicted + 2 std dev')
plt.plot(mean_minus_2_std,color='blue',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(mean_plus_2_std,color='green',linewidth=0.6,label='Predicted - 2 std dev')
plt.plot(mean_minus_2_std2[0],color='red',linewidth=0.6,label='Predicted + 2 std dev')
plt.plot(mean_plus_2_std2[0],color='red',linewidth=0.6,label='Predicted - 2 std dev')
plt.legend()
plt.show()

