import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, activation='leaky_relu'):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(num_channels, (3, 3), strides=1, padding='same', use_bias=False)
        self.norm1 = BatchNormalization()
        self.activation = Activation(activation)
        self.conv2 = Conv2D(num_channels, (3, 3), strides=1, padding='same', use_bias=False)
        self.norm2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        # Skip connection
        return Add()([inputs, x])


class ResNet(Model):
    def __init__(self, num_input_channels, num_output_channels, num_blocks, num_channels, 
                 need_residual=True, activation='leaky_relu', need_sigmoid=True, pad='same'):
        super(ResNet, self).__init__()

        self.initial_conv = Conv2D(num_channels, (3, 3), strides=1, padding=pad, use_bias=True)
        self.activation = Activation(activation)
        self.residual_blocks = []

        # Add residual blocks
        for _ in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(num_channels, activation))

        self.final_conv1 = Conv2D(num_channels, (3, 3), strides=1, padding=pad, use_bias=False)
        self.final_norm = BatchNormalization()
        self.final_conv2 = Conv2D(num_output_channels, (3, 3), strides=1, padding=pad, use_bias=True)

        if need_sigmoid:
            self.sigmoid = Activation('sigmoid')
        else:
            self.sigmoid = None

    def call(self, inputs):
        x = self.initial_conv(inputs)
        x = self.activation(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.final_conv1(x)
        x = self.final_norm(x)
        x = self.final_conv2(x)

        if self.sigmoid:
            x = self.sigmoid(x)
        return x
