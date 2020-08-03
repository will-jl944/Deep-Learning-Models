from config import config
from utils.tf_utilities import *
import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, scope='UNET'):
        super(UNet, self).__init__(name=scope)

        self.scope = scope
        with tf.name_scope(self.scope):
            # /8
            base = 4
            # Encode
            self.conv1_1 = Conv2d(1, base, scope='conv1_1', kernal_size=3, stride=1)
            self.conv1_2 = Conv2d(base, base, scope='conv1_2', kernal_size=3, stride=1)
            # self.mp1 = tf.keras.layers.MaxPool2D()

            self.conv2_1 = Conv2d(base, 2*base, scope='conv2_1', kernal_size=3, stride=1)
            self.conv2_2 = Conv2d(2*base, 2*base, scope='conv2_2', kernal_size=3, stride=1)
            # self.mp2 = tf.keras.layers.MaxPool2D()

            self.conv3_1 = Conv2d(2*base, 4*base, scope='conv3_1', kernal_size=3, stride=1)
            self.conv3_2 = Conv2d(4*base, 4*base, scope='conv3_2', kernal_size=3, stride=1)
            # self.mp3 = tf.keras.layers.MaxPool2D()

            self.conv4_1 = Conv2d(4*base, 8*base, scope='conv4_1', kernal_size=3, stride=1)
            self.conv4_2 = Conv2d(8*base, 8*base, scope='conv4_2', kernal_size=3, stride=1)
            # self.mp4 = tf.keras.layers.MaxPool2D()

            self.conv5_1 = Conv2d(8*base, 16*base, scope='conv5_1', kernal_size=3, stride=1)
            self.conv5_2 = Conv2d(16*base, 16*base, scope='conv5_2', kernal_size=3, stride=1)

            # Decode
            # up-conv
            # self.up1 = tf.keras.layers.UpSampling2D()
            self.conv6_1 = Conv2d(16*base, 8*base, scope='conv6_1', kernal_size=2, stride=1)
            # concat
            # concat1 = tf.keras.layers.concatenate()
            self.conv6_2 = Conv2d(16*base, 8*base, scope='conv6_2', kernal_size=3, stride=1)
            self.conv6_3 = Conv2d(8*base, 8*base, scope='conv6_3', kernal_size=3, stride=1)

            # up-conv
            # self.up2 = tf.keras.layers.UpSampling2D()
            self.conv7_1 = Conv2d(8*base, 4*base, scope='conv7_1', kernal_size=2, stride=1)
            # concat
            # concat2 = tf.keras.layers.concatenate()
            self.conv7_2 = Conv2d(8*base, 4*base, scope='conv7_2', kernal_size=3, stride=1)
            self.conv7_3 = Conv2d(4*base, 4*base, scope='conv7_3', kernal_size=3, stride=1)

            # up-conv
            # self.up3 = tf.keras.layers.UpSampling2D()
            self.conv8_1 = Conv2d(4*base, 2*base, scope='conv8_1', kernal_size=2, stride=1)
            # concat
            # concat3 = tf.keras.layers.concatenate()
            self.conv8_2 = Conv2d(4*base, 2*base, scope='conv8_2', kernal_size=3, stride=1)
            self.conv8_3 = Conv2d(2*base, 2*base, scope='conv8_3', kernal_size=3, stride=1)

            # up-conv
            # self.up4 = tf.keras.layers.UpSampling2D()
            self.conv9_1 = Conv2d(2*base, base, scope='conv9_1', kernal_size=2, stride=1)
            # concat
            # concat3 = tf.keras.layers.concatenate()
            self.conv9_2 = Conv2d(2*base, base, scope='conv9_2', kernal_size=3, stride=1)
            self.conv9_3 = Conv2d(base, base, scope='conv9_3', kernal_size=3, stride=1)

            self.conv10 = Conv2d(base, 2, scope='conv10', kernal_size=3, stride=1)
            self.final = Conv2d(2, 1, scope='final', kernal_size=1, stride=1, padding='VALID', activation='sigmoid')


    def __call__(self, inputs, training=True):
        with tf.name_scope(self.scope):
            input1 = inputs
            output1 = self.conv1_1(input1)
            output1 = self.conv1_2(output1)
            # tf.print('output1: ', output1, {'min': tf.math.reduce_min(output1)})

            input2 = tf.keras.layers.MaxPool2D()(output1)
            output2 = self.conv2_1(input2)
            output2 = self.conv2_2(output2)
            # tf.print('output12: ', output2)

            input3 = tf.keras.layers.MaxPool2D()(output2)
            output3 = self.conv3_1(input3)
            output3 = self.conv3_2(output3)
            # tf.print('output3: ', output3)

            input4 = tf.keras.layers.MaxPool2D()(output3)
            output4 = self.conv4_1(input4)
            output4 = self.conv4_2(output4)
            # tf.print('output4: ', output4)

            input5 = tf.keras.layers.MaxPool2D()(output4)
            output5 = self.conv5_1(input5)
            output5 = self.conv5_2(output5)
            # tf.print('output5: ', output5)

            input6 = self.conv6_1(tf.keras.layers.UpSampling2D()(output5))
            input6 = tf.keras.layers.concatenate([output4, input6], axis=3)
            output6 = self.conv6_2(input6)
            output6 = self.conv6_3(output6)
            # tf.print('output6: ', output6)

            input7 = self.conv7_1(tf.keras.layers.UpSampling2D()(output6))
            input7 = tf.keras.layers.concatenate([output3, input7], axis=3)
            output7 = self.conv7_2(input7)
            output7 = self.conv7_3(output7)
            # tf.print('output7: ', output7)

            input8 = self.conv8_1(tf.keras.layers.UpSampling2D()(output7))
            input8 = tf.keras.layers.concatenate([output2, input8], axis=3)
            output8 = self.conv8_2(input8)
            output8 = self.conv8_3(output8)
            # tf.print('output8: ', output8)

            input9 = self.conv9_1(tf.keras.layers.UpSampling2D()(output8))
            input9 = tf.keras.layers.concatenate([output1, input9], axis=3)
            output9 = self.conv9_2(input9)
            output9 = self.conv9_3(output9)
            # tf.print('output9: ', output9)

            output10 = self.conv10(output9)
            final_output = self.final(output10)
            # tf.print('final: ', final_output)

            return final_output
