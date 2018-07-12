# Author: Abdulrahman Tolba
# Date: 7/9/2018
# Name: neuralNetwork.py


import numpy as np                                                              # Import NumPy Library
import tensorflow as tf                                                         # Import TensorFlow Deep Learning Library
import matplotlib.pyplot as plt                                                 # Import matplotlib.pyplot Library to display graphs

np.random.seed(101)                                                             # Generate random Numpy
tf.set_random_seed(101)                                                         # and TensorFlow seeders

rand_a = np.random.uniform(0, 100, (5,5))                                       # Generate a 5x5 array with random numbers between 0-100
rand_b = np.random.uniform(0, 100, (5,1))                                       # Generate a 5x1 array with random numbers between 0-100

a = tf.placeholder(tf.float32)                                                  # Initialize empty placeholders for a & b
b = tf.placeholder(tf.float32)                                                  # to feed training data into later

add_op = tf.add(a,b)                                                            # Addition function that adds two placeholders
mul_op = tf.multiply(a,b)                                                       # Multiplication function that multiplies two placeholders

with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={a:rand_a,b:rand_b})                # Use the add_op variable to add matrices rand_a and rand_b and store the result in add_result
    print(add_result)
    print("\n")
    mult_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})              # Use the mul_op variable to multiply matrices rand_a and rand_b and store the result in mult_result
    print(mult_result)

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))                              # Input data
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))                 # Weight
b = tf.Variable(tf.ones([n_dense_neurons]))                                     # Bias

xW = tf.matmul(x,W)                                                             # tf.matmul(x,W) = x * w
z = tf.add(xW, b)                                                               # tf.add(xW,b) = z = x*w + b

a = tf.sigmoid(z)                                                               # Activation function (sigmoid)
init = tf.global_variables_initializer()                                        # Initialization function

with tf.Session() as sess:
    sess.run(init)                                                              # Initialize variables before using them
    layer_out = sess.run(a, feed_dict={x:np.random.random([1,n_features])})

print(layer_out)

# Simple Regression Example
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)                  # np.linspace will give us a linear graph, np.random.uniform adds noise to it
y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)

plt.plot(x_data, y_label, '*')                                                  # Plot the values on a graph
plt.legend()
plt.show()                                                                      # Show the graph in a new window

m = tf.Variable(0.44)                                                           # Slope (randomly generated number)
b = tf.Variable(0.87)                                                           # Y-Intercept (randomly generated number)

error = 0
for x,y in zip(x_data,y_label):                                                 # For loop that gets x & y values from x_data and y_label
    y_hat = m*x + b                                                             # Predicted value
    error += (y-y_hat)**2                                                       # Compare y to y_hat and square it to get the error (cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)              # Optimize the model using gradient descent
train = optimizer.minimize(error)                                               # Train the model using the error/cost 
init = tf.global_variables_initializer()                                        # Initialize all variables

with tf.Session() as sess:
    sess.run(init)
    training_steps = 50                                                         # Train the model <training_steps> times
    for i in range(training_steps):
        sess.run(train)
    final_slope, final_intercept = sess.run([m,b])                              # Once the training is done, get the final values of m and b

x_test = np.linspace(-1,11,10)
y_pred_plot =  final_slope * x_test + final_intercept                           # Y = mx + b

plt.plot(x_test, y_pred_plot, 'r')                                              # Plot the line of the graph using y_pred_plot
plt.plot(x_data, y_label, '*')                                                  # Plot the points on the graph
plt.legend()   
plt.show()                                                                      # Display the graph in a new window