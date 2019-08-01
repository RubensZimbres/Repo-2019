import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X = tf.placeholder(dtype=tf.float32)
X2 = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)

num_hidden=128


# Build a hidden layer Left
W_hidden = tf.Variable(tf.random.normal([784, num_hidden]))
b_hidden = tf.Variable(tf.random.normal([num_hidden]))
p_hidden = tf.nn.relu( tf.add(tf.matmul(X, W_hidden), b_hidden) )

W_hidden2 = tf.Variable(tf.random.normal([num_hidden, num_hidden]))
b_hidden2 = tf.Variable(tf.random.normal([num_hidden]))
p_hidden2 = tf.nn.relu( tf.add(tf.matmul(p_hidden, W_hidden2), b_hidden2) )

# Build a hidden layer Right
W_hiddenR = tf.Variable(tf.random.normal([784, num_hidden]))
b_hiddenR = tf.Variable(tf.random.normal([num_hidden]))
p_hiddenR = tf.nn.relu( tf.add(tf.matmul(X2, W_hiddenR), b_hiddenR) )

W_hidden2R = tf.Variable(tf.random.normal([num_hidden, num_hidden]))
b_hidden2R = tf.Variable(tf.random.normal([num_hidden]))
p_hidden2R = tf.nn.relu( tf.add(tf.matmul(p_hiddenR, W_hidden2R), b_hidden2R) )

# Conncatenate Left + Right
W_concat = tf.Variable(tf.random.normal([num_hidden,num_hidden]))
b_concat = tf.Variable(tf.random.normal([num_hidden]))
p_concat2 = tf.nn.relu(tf.add(tf.matmul(tf.math.add(p_hidden2,p_hidden2R), W_concat), b_concat))

# Build the output layer
W_output = tf.Variable(tf.random.normal([num_hidden, 10]))
b_output = tf.Variable(tf.random.normal([10]))
p_output = tf.nn.softmax( tf.add(tf.matmul(p_concat2, W_output), b_output))

loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=Y,predictions=p_output))
accuracy=1-tf.sqrt(loss)

minimization_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
saver = tf.train.Saver()

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

feed_dict = {
    X: norm(x_train[0:1000].reshape(-1,784)).astype(np.float32),
    X2: norm(x_train[0:1000].reshape(-1,784)).astype(np.float32),
    Y: pd.get_dummies(y_train[0:1000])
}

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(8000):
        J_value = session.run(loss, feed_dict)
        acc = session.run(accuracy, feed_dict)
        if step % 100 == 0:
            print("Step:", step, " Loss:", J_value," Accuracy:", acc)
            session.run(minimization_op, feed_dict)
