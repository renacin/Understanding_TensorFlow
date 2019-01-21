# Name:                                             Renacin Matadeen
# Date:                                                01/20/2018
# Title                                             NN Example, MPG
#
#
# ----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------------------------------------------------
"""
    Notes:
        + From Keras Tutorial
            - https://www.tensorflow.org/tutorials/keras/basic_regression

        + Misc
            - print(model.summary())
# """
# ----------------------------------------------------------------------------------------------------------------------

# Import Car_Data
raw_data_df = pd.read_csv("C:/Users/renac/Documents/Programming/Python/Tensorflow/Understanding_TensorFlow/Data/Cars_Data.csv")
all_data = raw_data_df.copy()

# Remove "?" from dataset
all_data = all_data[(all_data != '?').all(axis=1)]

# Focus On Everything But MPG, & Origin
data = all_data[["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration"]]

# Normalize Data | Max Score Standardization | Skip MPG & Origin
data_max = data.max(axis=0)
data_max_list = list(data_max.values)
data = data.astype(float) / data_max.astype(float)

# Add Origin, But In OneHot Encoding
origin = all_data[["Origin"]]
data['USA'] = (origin == 1)*1.0
data['Europe'] = (origin == 2)*1.0
data['Japan'] = (origin == 3)*1.0

# Add MPG To Dataframe
data["Mpg"] = all_data[["Mpg"]]

# Separate Into Train & Test Datasets
train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)

# Get Statistics
train_stats = train_dataset.describe().transpose()

# Create Labels
train_labels = train_dataset.pop('Mpg')
test_labels = test_dataset.pop('Mpg').tolist()

# ----------------------------------------------------------------------------------------------------------------------
# Build Model
def build_model():
  model = keras.Sequential([
    layers.Dense(50, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(50, activation=tf.nn.relu),
    layers.Dense(50, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.AdamOptimizer(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

  return model

# Fit Model
model = build_model()
history = model.fit(train_dataset, train_labels, epochs=1000, validation_split = 0.2, verbose=0)
hist = pd.DataFrame(history.history)

# Make Predictions
test_predictions = model.predict(test_dataset).flatten()

# Find R Squared
r_squared = round(r2_score(test_labels, test_predictions), 4)

if r_squared < 0:
    print("R-Squared: Poor Model")
else:
    print("R-Squared: {0}".format(r_squared))

# Save The Model
path = "C:/Users/renac/Documents/Programming/Python/Tensorflow/Understanding_TensorFlow/Model_Saves/"

try:
    model_json = model.to_json()
    with open(path + "CarMPG_Model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("CarMPG_Model.h5")
    print("Saved Model")
except:
    print("Error Could Not Save Model")
# ----------------------------------------------------------------------------------------------------------------------
