# Author: Abdulrahman Tolba
# Date: 7/10/2018
# Name: regressionModel.py

''' Sources: 
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
    https://pandas.pydata.org/pandas-docs/stable/dsintro.html
'''

import numpy as np                                                      # Import NumPy Library
import pandas as pd                                                     # Import Pandas Library
import matplotlib.pyplot as plt                                         # Import MatPlotLib.pyplot Library
import tensorflow as tf                                                 # Import TensorFlow Deep Learning Library

x_data = np.linspace(0.0,10.0,1000000)                                  # Generate a million points between 0.0-10.0
noise = np.random.randn(len(x_data))                                    # Generate some noise using the length of x_data

y_true = (0.5 * x_data) + 5 + noise                                     # y = mx + b, with added noise to it
                                                                        #     m = 0.5

x_df = pd.DataFrame(data=x_data, columns=['X Data'])                    # X and Y Dataframes (2D tabular data structures
y_df = pd.DataFrame(data=y_true, columns=['Y'])                         # with labeled axes)  
y_df.head()

my_data = pd.concat([x_df, y_df], axis=1)
my_data.sample(n = 250).plot(kind='scatter',x='X Data',y='Y')           # Plot only random samples, not all 1,000,000 points

# When training a lot of complex models, you cant just feed in millions of data at once, so you feed it batches of data instead

batch_size = 8                                                          # Batch size
m = tf.Variable(0.81)                                                   # Slope variable (randomly generated)
b = tf.Variable(0.17)                                                   # Y-intercept variable (randomly generated)

xph = tf.placeholder(tf.float32, [batch_size])                          # Initialize empty placeholders for x & y
yph = tf.placeholder(tf.float32, [batch_size])                          # to feed training data into later

y_model = m * xph + b                                                   # y = mx + b

error = tf.reduce_sum(tf.square(yph-y_model))                           # Compare y to y_hat and square it to get the error (cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)      # Optimize the model using gradient descent
train = optimizer.minimize(error)                                       # Train the model using the error/cost (minimize the error)

init = tf.global_variables_initializer()                                # Initialize all variables

with tf.Session() as sess:
    sess.run(init)                                                      # Run the initializer
    batches = 10000                                                     # Feed in 10000 batches of data, each batch has 8 corresponding data points
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)      # Corresponds to a 8 random indices from 0 - len(x_data)
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}              # Feed dictionary that feeds data to xph and yph
        sess.run(train,feed_dict=feed)                                  # Training the model with minimal cost and feeding the input data
    model_m, model_b = sess.run([m,b])

y_hat = x_data * model_m + model_b                                      # y = m*x + b   where y = y_hat, m = model_m, x = x_data, and b = model_b
my_data.sample(250).plot(kind='scatter',x='X Data', y='Y')              # Generate a scatter plot with 250 random points
plt.plot(x_data,y_hat, 'r')                                             # Display the plot
plt.show()

feat_cols = [ tf.feature_column.numeric_column('x',shape=[1])]          # Single feature numeric columns
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)     # Train a linear regression model to predict label value given observation of feature values.


from sklearn.model_selection import train_test_split                    # Import train_test_split from the SciKit Library

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, 
                                    test_size=0.2, random_state=101)    # Split x_data into random train and test subsets


# Estimator requirements: an input function, training function, and evaluation (test) function.

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, 
            y_train, batch_size=8, num_epochs=None, shuffle=True)       # Input function
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, 
            y_train, batch_size=8, num_epochs=1000, shuffle=False)      # Training function
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval}, 
            y_eval, batch_size=8, num_epochs=1000, shuffle=False)       # Evaluation function

estimator.train(input_fn = input_func, steps = 1000)                    # Train the estimator model using the input function, which takes 1000 steps at a time
train_metrics = estimator.evaluate(input_fn = train_input_func, 
                                                    steps=1000)         # Evaluate the training metrics using the training function
eval_metrics = estimator.evaluate(input_fn=eval_input_func, 
                                                    steps=1000)         # Evaluate the test metrics using the input function

print("Training Data Metrics:")
print(train_metrics)

print("Evauation Metrics")
print(eval_metrics)
 
brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

my_data.sample(n=250).plot(kind='scatter',x='X Data', y='Y')
plt.plot(brand_new_data,predictions,'r')
plt.show()