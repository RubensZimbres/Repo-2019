#!pip install tf-nightly-gpu-2.0-preview --user

from tensorflow.keras.layers import Layer
import tensorflow as tf


class Linear(Layer):
  """y = w.x + b"""

  def __init__(self, units=32):
      super(Linear, self).__init__()
      self.units = units

  def build(self, input_shape):
      self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
      self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

  def call(self, inputs):
      return tf.matmul(inputs, self.w) + self.b


# Instantiate our lazy layer.
linear_layer = Linear(6)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
y


''' DATASET '''

# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# Instantiate our linear layer (defined above) with 10 units.
linear_layer = Linear(10)

# Instantiate a logistic loss function that expects integer targets.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# Iterate over the batches of the dataset.

'''TRAIN'''

for step, (x, y) in enumerate(dataset):
  
  # Open a GradientTape.
    with tf.GradientTape() as tape:

    # Forward pass.
        logits = linear_layer(x)

    # Loss value for this batch.
        loss = loss_fn(y, logits)
     
  # Get gradients of weights wrt the loss.
    gradients = tape.gradient(loss, linear_layer.trainable_weights)
  
  # Update the weights of our linear layer.
    optimizer.apply_gradients(zip(gradients, linear_layer.trainable_weights))
  
  # Logging.
    if step % 100 == 0:
        print(step, float(loss))
        
###################################################################################################        
        
        
'''MLP'''

class MLP(Layer):
    """Simple stack of Linear layers."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)

mlp = MLP()


(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype('float32') / 255, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(64)

# Instantiate a logistic loss function that expects integer targets.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)


for i in range(0,500):
    for x, y in dataset:
      
      # Open a GradientTape.
        with tf.GradientTape() as tape:
    
        # Forward pass.
            logits = mlp(x)
    
        # Loss value for this batch.
            loss = loss_fn(y, logits)
         
      # Get gradients of weights wrt the loss.
        gradients = tape.gradient(loss, mlp.trainable_weights)
      
      # Update the weights of our linear layer.
        optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))
      
      # Logging.
        if step % 100 == 0:
            print(step, float(loss))

        #print(y-tf.cast(tf.argmax(logits,axis=1),tf.uint8))
    
    
    
