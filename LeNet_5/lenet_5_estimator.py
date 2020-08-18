import tensorflow as tf
import numpy as np


class lenet5(object):

    def __init__(self):
        pass

    def convlayer(self):
        self.parameters = []

        with tf.name_scope("conv1_1") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 6], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[6], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.maxpool1_1 = tf.nn.max_pool(self.conv1_1,
                                    pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding="SAME",
                                    )

        with tf.name_scope("conv1_2") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 16], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.maxpool1_1, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.maxpool1_2 = tf.nn.max_pool(self.conv1_1,
                                    pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding="SAME",
                                    )

        with tf.name_scope("conv1_3") as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 120], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.maxpool1_2, kernel, [1, 1, 1, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

    def fc_layer(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.conv1_3.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 84],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[84], dtype=tf.float32),
                               trainable=True, name='biases')
            conv1_3_flat = tf.reshape(self.conv1_3, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(conv1_3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([84, 10],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.parameters += [fc2w, fc2b]