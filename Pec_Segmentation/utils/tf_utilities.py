import tensorflow as tf
import numpy as np
from config import config


def activation_func(z, activation='leaky_relu'):
    if activation == 'leaky_relu':
        return tf.nn.leaky_relu(z)
    elif activation == 'relu':
        return tf.nn.relu(z)
    elif activation == 'selu':
        return tf.nn.selu(z)
    elif activation == 'tanh':
        return tf.nn.tanh(z)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(z)
    elif activation =='linear':
        return z

    assert False, 'Activation Func "{}" not Found'.format(activation)


class Dense(tf.keras.layers.Layer):
    def __init__(self, input_shape, output_features, new_layer_gain=np.sqrt(2), scope='Dense'):
        super(Dense, self).__init__(name=scope)
        self.scope = scope

        initializer = tf.keras.initializers.VarianceScaling(scale=new_layer_gain, mode='fan_in', distribution='truncated_normal', seed=None)
        self.weight = tf.Variable(name='weights', shape=[input_shape, output_features], initial_value=initializer([input_shape, output_features]))
        self.biases = tf.Variable(name='bias', shape=[output_features], initial_value=initializer([output_features]))

    def __call__(self, x):
        with tf.name_scope(self.scope):

            y = tf.matmul(x, self.weight) + self.biases
            return tf.nn.relu(y)


class Conv2d(tf.keras.layers.Layer):
    def __init__(self, num_channel_in, num_channel_out, stride=1, padding='SAME', activation='leaky_relu', kernal_size=3, scope='Conv2d'):
        super(Conv2d, self).__init__(scope)
        self.scope = scope

        with tf.name_scope(self.scope):

            self.stride = stride
            self.padding = padding
            self.activation = activation

            #with tf.device("/device:{}:0".format(controller)):
            if isinstance(kernal_size, int):
                kernal_height = kernal_size
                kernal_width = kernal_size
            else:
                kernal_height = kernal_size[0]
                kernal_width = kernal_size[1]

            initializer = tf.random_normal_initializer()

            kernal_shape = [kernal_height, kernal_width, int(num_channel_in), int(num_channel_out)]
            bias_shape = [int(num_channel_out)]

            self.kernel = tf.Variable(name='weights', shape=kernal_shape, initial_value=initializer(kernal_shape), dtype=tf.float32)
            self.biases = tf.Variable(name='bias', shape=bias_shape, initial_value=initializer(bias_shape), dtype=tf.float32)

    def __call__(self, input_vol):
        with tf.name_scope(self.scope):

            conv = tf.add(tf.nn.conv2d(input_vol, self.kernel, [1, self.stride, self.stride, 1], padding=self.padding), self.biases)

            return activation_func(conv, self.activation)

