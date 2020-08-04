from config import config
from utils.tf_utilities import *
import tensorflow as tf


class Feature_Extractor(tf.keras.Model):
    def __init__(self, num_channel_in, scope='Feature Extractor'):
        super(Feature_Extractor, self).__init__()
        self.scope = scope
        with tf.name_scope(self.scope):
            base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
            for layer in base_model.layers:
                layer.trainable = False
            self.extractor = tf.keras.Model(base_model.input, outputs=base_model.get_layer('block_15_add').output)
            self.extractor.trainable = False

    def __call__(self, inputs, training=True):
        with tf.name_scope(self.scope):
            return self.extractor(inputs)


class Regressor(tf.keras.layers.Layer):
    def __init__(self, num_channel_in, scope='Regressor'):
        self.scope =scope
        super(Regressor, self).__init__()
        self.layers = []

        base = 32  # 256, 256, 128, 128, 64, 64

        self.layers.append(tf.keras.layers.MaxPool2D())

        self.layers.append(tf.keras.layers.Dropout(rate=.5))

        self.layers.append(Conv2d(num_channel_in, base, scope='conv1_1', kernal_size=3, stride=1))
        self.layers.append(Conv2d(base, base/2, scope='conv1_2', kernal_size=1, stride=1))

        self.layers.append(Conv2d(base/2, base, scope='conv2_1', kernal_size=3, stride=1))
        self.layers.append(Conv2d(base, base/2, scope='conv2_2', kernal_size=1, stride=1))

        self.layers.append(Conv2d(base/2, base, scope='conv3_1', kernal_size=3, stride=1))

        self.layers.append(Conv2d(base, 3, scope='conv4_1', kernal_size=[8, 4], stride=1, activation='sigmoid', padding='VALID'))

    def __call__(self, inputs, training=True):
        with tf.name_scope(self.scope):
            blocks = [inputs]

            for layer in self.layers:
                blocks.append(layer(blocks[-1]))

            return blocks[-1]


class Detector(tf.keras.Model):
    def __init__(self, scope='Detector'):
        super(Detector, self).__init__(name=scope)

        self.scope = scope
        with tf.name_scope(self.scope):
            base_model = tf.keras.applications.MobileNetV2(input_shape=[512, 256, 3], include_top=False, weights='imagenet')
            for layer in base_model.layers:
                layer.trainable = False
            self.feature_extractor = tf.keras.Model(base_model.input, outputs=base_model.get_layer('block_15_add').output)
            self.feature_extractor.trainable = False

            self.regressor = Regressor(num_channel_in=160)

    def __call__(self, inputs, training=True):
        with tf.name_scope(self.scope):
            inputs = tf.cast(inputs, tf.float32)

            inputs = tf.tile(inputs, multiples=[1, 1, 1, 3])

            features = self.feature_extractor(inputs)
            res = self.regressor(features)

            # tf.print(res)
            return res
