""" Self made module by @ Z.Han"""

""" ResBlock.py module contains 3*3 ResNet Block for ResNet designing, 
parameters are imported as kernel size, filters for instance 
class attribution, call function could import the input tensor 
automatically
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
class ResnetBlock(Model):

    def __init__(self, kernel_size, filters):
        reg = 1e-3
        trainable = False
        super(ResnetBlock, self).__init__(name='') #name='Res'
        self.filters1, self.filters2 = filters

        self.conv2a = layers.Conv2D(self.filters1, (1, 1), kernel_regularizer=regularizers.l2(reg))
        self.bn2a = layers.BatchNormalization(trainable = trainable)
        # self.gn2a = tfa.layers.GroupNormalization(groups=5, axis=3)

        self.padding = layers.ZeroPadding2D(padding=(1, 1))
        self.conv2b = layers.Conv2D(self.filters2, kernel_size, kernel_regularizer=regularizers.l2(reg))
        self.bn2b = layers.BatchNormalization(trainable = trainable)
        # self.gn2b = tfa.layers.GroupNormalization(groups=5, axis=3)

        self.conv2c = layers.Conv2D(self.filters2, kernel_size, kernel_regularizer=regularizers.l2(reg))
        self.bn2c = layers.BatchNormalization(trainable = trainable)
        # self.gn2c = tfa.layers.GroupNormalization(groups=5, axis=3)

    def call(self, tensor):
        if self.filters1 != 0:
            tensor = self.conv2a(tensor)

        x = self.padding(tensor)
        x = self.conv2b(x)
        x = self.bn2b(x)
        x = tf.nn.relu(x)

        x = self.padding(x)
        x = self.conv2c(x)


        x += tensor

        x = self.bn2c(x)

        return tf.nn.relu(x) #leaky_relu