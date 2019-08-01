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
class BahdanauAttention(Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(values, 1)
    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector


class MLP(Layer):
    """Simple stack of Linear layers."""

    def __init__(self):
        super(MLP, self).__init__()
        self.linear_1 = Linear(32)
        self.attention=BahdanauAttention(10)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(10)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.attention(x)
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



for step, (x, y) in enumerate(dataset):
  
  # Open a GradientTape.
    for i in range(0,30):
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

    print(y-tf.cast(tf.argmax(logits,axis=1),tf.uint8))
    
    
    
