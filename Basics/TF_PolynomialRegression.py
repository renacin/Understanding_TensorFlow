# Name:                                             Renacin Matadeen
# Date:                                                12/28/2018
# Title                           TensorFlow Example - Multivariate Polynomial Regression
#
#
# ----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

# ----------------------------------------------------------------------------------------------------------------------

def get_inner_data(x):
    list_ = []
    for x_ in x:
        for x__ in x_:
            x__ = round(x__, 5)
            list_.append(x__)
    return list_

def create_pred(frame, poly_degree, trainable_variables_list):
    # Derive Prediction_Y From Test Data!
    composite = 0
    for degree_ in range(1, poly_degree + 1):
        composite += (trainable_variables_list[degree_] * (frame ** degree_))

    # Add All The Columns Up Including The Bias
    prediction_column = 0
    for column in range(num_variables):
        prediction_column += composite.iloc[:, column]
    prediction_column += trainable_variables_list[0][0]

    return prediction_column

# ----------------------------------------------------------------------------------------------------------------------

# Define Hyper Parametres & Early Stop Limits
poly_degree = 5
learning_rate = 25
epochs = 5000

num_nochange_epochs = 10000
min_delta = 0.001

# ----------------------------------------------------------------------------------------------------------------------
"""
    NOTES:

        + To go from standardized coefficients to unstandardized coefficients divide Weight By XMAX

        + To make things easier just use the R Squared Calculator In SkLearn
            - It was nice to calculate R Squared by myself. But in the end it was quicker to use a SKLearn module

        + Use TensorBoard to visualize change?
            - Visualizing data is extremely benefitial!


# """
# ----------------------------------------------------------------------------------------------------------------------

# Get Data
df = pd.read_csv("C:/Users/renac/Documents/Programming/Python/Tensorflow/TensorFlow_PolynomialRegression/Data/Temp_Data.csv")

# Shuffle Data
data_df = df[["Time", "Temp"]]
data_df = df.sample(frac=1)

data_x = data_df[["Time"]]
data_y = data_df[["Temp"]]

# Number Of Observations
num_variables = len(data_x.columns)
num_observations = len(data_x)

# Implement Max Score Standardization
data_x_max = data_x.max(axis=0)
data_x_max_list = list(data_x_max.values)
data_x = (data_x / data_x_max)

# Split Data Into Train & Test Samples | 80:20 Split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.20, random_state=42)

# Version For R_Squared Calculation
train_x_ = train_x
train_y_ = train_y

# Turn To Values
x_train = np.array(train_x.values)
y_train = np.array(train_y.values)

# Numpy Create A Focus Array | Try To Be Efficient |
train_x = x_train.reshape(-1, num_variables)
train_y = y_train.reshape(-1, 1)

# Define Values For TensorFlow
X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

# ----------------------------------------------------------------------------------------------------------------------

# Instantiate Variables, In This Case You Need Two Variables Per Term - For Each Variable, Plus A Bias Term
Y_pred = tf.Variable(tf.random_normal([1, 1]), name='Bias')

for degree_ in range(1, poly_degree + 1):
    W = tf.Variable(tf.random_normal([num_variables, 1]), name="Degree_{0}_Weights".format(degree_))
    Y_pred = tf.add(tf.matmul(tf.pow(X, degree_), W), Y_pred)

# Define The Cost Function, Remember Mean Square Error!
eq = (tf.pow(Y_pred - Y, 2)) / (num_observations - 1)
cost = tf.reduce_sum(eq)

# Define The Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize All Values
init = tf.global_variables_initializer()

# Trainable Variable List
trainable_variables_list = []

# ----------------------------------------------------------------------------------------------------------------------

# Execute The Graph
with tf.Session() as sess:
    sess.run(init)

    # Early Stop Hyperparametres
    cost_storage = []
    num_epochs = 0

    # Loop Through Epochs
    for epoch in range(epochs):

        data_feed = {X: train_x, Y: train_y}
        sess.run(optimizer, feed_dict=data_feed)

        # Values For Early Stop
        c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
        cost_storage.append(c)

        # Initiate Early Stop Functionality
        if epoch > 0 and cost_storage[epoch-1] - cost_storage[epoch] > min_delta:
            num_epochs = 0
        else:
            num_epochs += 1

        if num_epochs > num_nochange_epochs:
            break

    # Collect Final Data
    c = sess.run(cost, feed_dict={X: train_x, Y: train_y})

    # Collect And Print Trainable Variables
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)

    for var, val in zip(tvars, tvars_vals):
        val = val.tolist()
        val = get_inner_data(val)
        trainable_variables_list.append(val)

    # Print Epoch Details
    print("Epoch: {0}, Cost: {1:.2f}".format(epoch + 1, c))

# ----------------------------------------------------------------------------------------------------------------------
# Error Statistics Based On Test Predictions & Test Data

# Derive Prediction_Y From Test Data!
prediction_column = create_pred(test_x, poly_degree, trainable_variables_list)

# Final Df W/ Predictions
pred_y_test = prediction_column.tolist()
pred_y_test = pd.DataFrame({'Y_Pred':pred_y_test})

# Accuracy Assessment RMSE
diff = (pred_y_test["Y_Pred"] - test_y["Temp"]) ** 2
diff_sum = diff.sum()
mean_diff = (diff_sum / len(test_y))
RMSE = (mean_diff)**(1/2)

RMSE = round(RMSE, 2)
print("RMSE [TEST DATA]: {0}".format(RMSE))

# ----------------------------------------------------------------------------------------------------------------------
# Error Statistics Based On Train Predictions & Train Data

# Derive Prediction_Y From Train Data!
prediction_column = create_pred(train_x_, poly_degree, trainable_variables_list)

# Final Df W/ Predictions
pred_y_train = prediction_column.tolist()
pred_y_train = pd.DataFrame({'Y_Pred':pred_y_train})

# Print R_Squared Value
r_squared = round(r2_score(train_y, pred_y_train), 4)

if r_squared < 0:
    print("R-Squared [TRAIN DATA]: Poor Model")
else:
    print("R-Squared [TRAIN DATA]: {0}".format(r_squared))


# ----------------------------------------------------------------------------------------------------------------------
# Turn Standardized Coefficients To Unstandardized Coefficients | Follow W/XMax^Degree
final_coefficients = [round(trainable_variables_list[0][0], 2)]
for degree_ in range(1, poly_degree + 1):
    temp_list = []
    for v, w in zip(data_x_max_list, trainable_variables_list[degree_]):
        un_std_coeff = w / (v**degree_)
        un_std_coeff = round(un_std_coeff, 4)
        temp_list.append(un_std_coeff)
    final_coefficients.append(temp_list)

print("COEFF: {0}".format(final_coefficients))

# ----------------------------------------------------------------------------------------------------------------------
# Create Predictions For All Data (100 percent Data Prediction)

# Derive Prediction_Y From All Data!
prediction_column = create_pred(data_x, poly_degree, trainable_variables_list)

# Final Df W/ Predictions
pred_y = prediction_column.tolist()
pred_y = pd.DataFrame({'Y_Pred':pred_y})

# Graph Data | Get A Better Understanding Of What's Going On
plt.title("Temperature Over Time")
plt.xlabel("Time [Hours]")
plt.ylabel("Temperature [Celcius]")
plt.plot(data_x, data_y, "+", label="Raw Data")
plt.plot(data_x, pred_y["Y_Pred"], "x", label="Fitted Data")

plt.grid(True)
plt.show()
