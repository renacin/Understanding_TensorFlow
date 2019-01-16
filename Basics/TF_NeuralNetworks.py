# Name:                                             Renacin Matadeen
# Date:                                                01/15/2018
# Title                                    TensorFlow Example - Neural Networks
#
#
# ----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# ----------------------------------------------------------------------------------------------------------------------
"""
    Neural Networks:
        + Input > Weight > Hidden Layer 1 > Activation Layer
        + Layer 1 > Weight > Hidden Layer 2 > Activation Layer
        + Etc...

        + Passing data straight through indicates a feed-forward network
        + We use backpropogation to modify the weights and biases to minimize the cost function
        + Feed-forward and backpropogation equals one epoch

        + Weights can't be zero, because it would never activate

        + tf.nn.relu (rectified linear) is your activation function

    General Notes:
        + Using MNIST Data
            - Each image is 28 X 28 Pixels
            - Therefore there are 784 inputs (features)
        + One_Hot refers to circuits
            - Indicates that one input is hot, and referes to one classification
        + Use batches when you can't load all data into memory

        + Tutorial From:
            - Youtube - Sentdex: Neural Network Model Part 3

# """
# ----------------------------------------------------------------------------------------------------------------------

def neural_network_model(data):

    # Dictionary Holds Weights, and Biases Of Respective Layer, Values Are Randomly Initialized
    hidden_1_layer = {"weights":tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))
                      }

    hidden_2_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl2]))
                      }

    hidden_3_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl3]))
                      }

    # What is the output layer? Depends. In this case the number of classes
    output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      "biases":tf.Variable(tf.random_normal([n_classes]))
                      }

    # This is the main computational graph of the model
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Initialze Session
    with tf.Session() as sess:
        sess.run(init)

        # Iterate Through Epochs
        for epoch in range(epochs):
            epoch_loss = 0

            # Use Batch Size To Chunk Data, and Loop Through Chunks
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # In Real Life you Must Develop Your Own Batch Chunking Function
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch: ", epoch + 1, "/", epochs, "| Loss: ", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))

# ----------------------------------------------------------------------------------------------------------------------

# Import MNIST Data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# Define Hyperparametres
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
learning_rate = 0.001
epochs = 25

# ----------------------------------------------------------------------------------------------------------------------

# Define Placeholders, and Shape
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

# Run Model
train_neural_network(x)
