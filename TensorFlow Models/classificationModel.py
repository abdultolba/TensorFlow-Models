# ******************************************************************************
# Author: Abdulrahman Tolba
# Date: 7/10/18
# Name:classificationModel.py
# ******************************************************************************

''' Sources:
    https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html                   TensorFlow Feature Columns
    https://www.tensorflow.org/api_docs/python/tf/feature_column                                            Python  Feature Columns
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.hist.html                       Pandas DataFrames
'''

import pandas as pd                                                                                         # Import Pandas Library
import tensorflow as tf                                                                                     # Import TensorFlow Library
import matplotlib.pyplot as plt                                                                             # Import MatPlotLib Library
from sklearn.model_selection import train_test_split                                                        # Import train_test_split from SkLearn Library

diabetes = pd.read_csv(r'C:\Users\atolba\Desktop\FULL-TENSORFLOW-NOTES-AND-DATA\
                        Tensorflow-Bootcamp-master\02-TensorFlow-Basics\pima-indians-diabetes.csv')         # Read a CSV file as input data
diabetes.head()                                                                                             # View the input dataset in a table

cols_to_norm = ['Number_pregnant', 'Glucose_concentration',
                'Blood_pressure', 'Triceps','Insulin', 'BMI', 'Pedigree']                                   # List of columns that need to be 'normalized'

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min() ))       # Normalize the columns in diabetes[] using a lambda function
diabetes.head()                                                                                             # View the input dataset in a table

num_preg = tf.feature_column.numeric_column('Number_pregnant')                                              # Initalize num_preg using the Number_pregnant column
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')                                     # Initialize plsma_gluc using the GLucose_concentration column
dias_press = tf.feature_column.numeric_column('Blood_pressure')                                             # Initialize dias_press using the Blood_pressure column
tricep = tf.feature_column.numeric_column('Triceps')                                                        # Initialize tricep using the Triceps column
insulin = tf.feature_column.numeric_column('Insulin')                                                       # Initialize insulin using the Insulin column
bmi = tf.feature_column.numeric_column('BMI')                                                               # Initialize bmi using the BMI column
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')                                            # Initialize diabetes_pedigree using the Pedigree column
age = tf.feature_column.numeric_column('Age')                                                               # Initialie age using the Age column

assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A','B', 'C', 'D'])    # Store the assigned group using an alphabtical categorical column
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

diabetes['Age'].hist(bins=20)                                                                               # Plot a histogram of the Age column with 20 bins
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])                    # Bucketize the Age column
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin,
             bmi, diabetes_pedigree, assigned_group, age_bucket]

# Train-Test Split
x_data = diabetes.drop('Class', axis=1)
x_data.head()

labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)        # Split x_data into random train and test subsets
input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train,
                                                 batch_size = 10, num_epochs = 1000, shuffle = True)         # Input function using x and y training data sets
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)                                 # Linear Classification model using tensorflow.estimator
model.train(input_fn = input_func, steps = 1000)                                                             # Training the model with the input function and 1000 steps
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test,
                                                      batch_size = 10, num_epochs = 1, shuffle = False)      # Evaluation function using x and y test data sets
results = model.evaluate(eval_input_func)

pred_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10,                           # Prediction inout function using x test data set
                                                      num_epochs = 1, shuffle = False)
predictions = model.predict(pred_input_func)                                                                 # Make a prediction using the prediction input function
my_pred = list(predictions) and                                                                              # Store prediction data into a python list

# ******************************************************************************
#                                 DNN model
# ******************************************************************************

dnn_model = tf.estimator.DNNClassifier(hidden_units = [10,20,20,20,10],
                                       feature_columns = feat_cols, n_classes = 2)
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin,
             bmi, diabetes_pedigree, embedded_group_col, age_bucket]
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train,
                                                 batch_size = 10, num_epochs = 1000, shuffle = True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_fn=input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
                                                      batch_size=10, num_epochs=1, shuffle=False)
dnn_model.evaluate(eval_input_func)
