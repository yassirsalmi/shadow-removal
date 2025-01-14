import tensorflow as tf
import numpy as np
import requests
from tensorflow.keras import layers, Model

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def conv3x3(filters, stride=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(filters, kernel_size=3, strides=stride,
                        padding='same', use_bias=False)

class BasicBlock(layers.Layer):
    def __init__(self, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(filters, stride)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(filters)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = None
        if stride != 1 or self.input_filters != filters:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])

    def build(self, input_shape):
        self.input_filters = input_shape[-1]
        super().build(input_shape)

    def call(self, x, training=False):
        residual = self.conv1(x)
        residual = self.bn1(residual, training=training)
        residual = tf.nn.relu(residual)
        
        residual = self.conv2(residual)
        residual = self.bn2(residual, training=training)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x, training=training)

        out = shortcut + residual
        out = tf.nn.relu(out)
        return out

def create_layer_basic(filters, blocks, stride=1):
    layers_list = [BasicBlock(filters, stride=stride)]
    for _ in range(blocks-1):
        layers_list.append(BasicBlock(filters, stride=1))
    return tf.keras.Sequential(layers_list)

class Resnet18(Model):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same',
                                 use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        
        self.layer1 = create_layer_basic(64, blocks=2, stride=1)
        self.layer2 = create_layer_basic(128, blocks=2, stride=2)
        self.layer3 = create_layer_basic(256, blocks=2, stride=2)
        self.layer4 = create_layer_basic(512, blocks=2, stride=2)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        feat8 = self.layer2(x, training=training)  # 1/8
        feat16 = self.layer3(feat8, training=training)  # 1/16
        feat32 = self.layer4(feat16, training=training)  # 1/32
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                wd_params.append(layer.kernel)
                if layer.use_bias:
                    nowd_params.append(layer.bias)
            elif isinstance(layer, layers.BatchNormalization):
                nowd_params.extend(layer.trainable_variables)
        return wd_params, nowd_params

if __name__ == "__main__":
    # Force TensorFlow to use CPU
    tf.config.set_visible_devices([], 'GPU')
    
    net = Resnet18()
    x = tf.random.normal((16, 224, 224, 3))  # Note: TF uses channels-last format
    out = net(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    net.get_params()
