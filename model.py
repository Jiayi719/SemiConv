import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Add, Input, Conv2D, BatchNormalization,
                                     Activation, MaxPooling2D)


# Define truncated ResNet50.
# No pooling and Conv2D strides for simplicity.
class ResNet50Trunc(object):
    """
    Reference:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    """

    def __init__(self, input_shape, output_dim=8, semi_conv=False):
        x = Input(shape=input_shape)
        h = Conv2D(64, kernel_size=(7, 7), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)

        h = Conv2D(output_dim, kernel_size=(1, 1), strides=(1, 1), padding='same')(h)
        if semi_conv:
            h = self._semi_conv(h)
        h = Conv2D(output_dim, kernel_size=(1, 1), strides=(1, 1), padding='same')(h)
        if semi_conv:
            h = self._semi_conv(h)
        y = Conv2D(output_dim, kernel_size=(1, 1), strides=(1, 1), padding='same')(h)
        if semi_conv:
            y = self._semi_conv(y)
        self.model = Model(x, y)

    def __call__(self, x):
        return self.model(x)

    def _semi_conv(self, x):
        # The semiconvolutional operator introduced by the paper.
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        channel = tf.shape(x)[3]
        u_x = tf.cast(tf.range(height), tf.float32) / tf.cast(height, tf.float32)
        u_x = tf.reshape(u_x, (1, height, 1, 1))
        u_x = tf.tile(u_x, (1, 1, width, 1))
        u_y = tf.cast(tf.range(width), tf.float32) / tf.cast(width, tf.float32)
        u_y = tf.reshape(u_y, (1, 1, width, 1))
        u_y = tf.tile(u_y, (1, height, 1, 1))
        u_other = tf.zeros_like(x)[..., :-2]
        u = tf.concat((u_x, u_y, u_other), 3)
        return x + u

    def _building_block(self, x, channel_out=256):
        channel = channel_out // 4
        h = Conv2D(channel, kernel_size=(1, 1), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel, kernel_size=(3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel_out, kernel_size=(1, 1), padding='same')(h)
        h = BatchNormalization()(h)
        shortcut = self._shortcut(x, output_shape=h.get_shape().as_list())
        h = Add()([h, shortcut])
        return Activation('relu')(h)

    def _shortcut(self, x, output_shape):
        input_shape = x.get_shape().as_list()
        channel_in = input_shape[-1]
        channel_out = output_shape[-1]

        if channel_in != channel_out:
            return self._projection(x, channel_out)
        else:
            return x

    def _projection(self, x, channel_out):
        return Conv2D(channel_out, kernel_size=(1, 1), padding='same')(x)
