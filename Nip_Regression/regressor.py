from config import config
from utils.tf_utilities import *
import tensorflow as tf


class Regressor(tf.keras.Model):
    def __init__(self, scope='REGRESSOR'):
        super(Regressor, self).__init__(name=scope)

        self.scope = scope
        with tf.name_scope(self.scope):
            base = 8  # 8, 8, 16, 16, 32, 32, 64, 64
            self.net_layers = []

            self.net_layers.append(Conv2d(1, base, scope='conv1_1', kernal_size=3, stride=1))
            self.net_layers.append(Conv2d(base, base, scope='conv1_2', kernal_size=3, stride=2))

            self.net_layers.append(Conv2d(base, 2 * base, scope='conv2_1', kernal_size=3, stride=1))
            self.net_layers.append(Conv2d(2 * base, 2 * base, scope='conv2_2', kernal_size=3, stride=2))

            self.net_layers.append(Conv2d(2 * base, 4 * base, scope='conv3_1', kernal_size=3, stride=1))
            self.net_layers.append(Conv2d(4 * base, 4 * base, scope='conv3_2', kernal_size=3, stride=2))

            self.net_layers.append(Conv2d(4 * base, 8 * base, scope='conv4_1', kernal_size=3, stride=1))
            self.net_layers.append(Conv2d(8 * base, 8 * base, scope='conv4_2', kernal_size=3, stride=2))

            self.net_layers.append(tf.keras.layers.Flatten())

            self.net_layers.append(Dense(input_shape=32*16*64, output_features=2))

    def __call__(self, inputs, training=True):
        with tf.name_scope(self.scope):
            blocks = [inputs]

            for layer in self.net_layers:
                blocks.append(layer(blocks[-1]))

            return blocks[-1]

