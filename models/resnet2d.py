import tensorflow as tf
from tensorflow.keras import layers, Model

def conv3x3(filters, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same",
                         dilation_rate=dilation, use_bias=False, kernel_initializer="he_normal")

def conv1x1(filters, stride=1):
    """1x1 convolution"""
    return layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, kernel_initializer="he_normal")

class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, filters, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3(filters, stride=stride)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(filters)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, filters, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        width = filters
        self.conv1 = conv1x1(width)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = conv3x3(width, stride=stride)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = conv1x1(filters * self.expansion)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.downsample = downsample

    def call(self, x, training=False):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(Model):
    def __init__(self, block, layers, num_classes=1000, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.inplanes = 64
        self.conv1 = layers.Conv2D(self.inplanes, kernel_size=7, strides=2, padding="same",
                                   use_bias=False, kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation="sigmoid", kernel_initializer="he_normal")

    def _make_layer(self, block, filters, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != filters * block.expansion:
            downsample = tf.keras.Sequential([
                conv1x1(filters * block.expansion, stride),
                layers.BatchNormalization()
            ])

        layers_list = [block(filters, stride=stride, downsample=downsample)]
        self.inplanes = filters * block.expansion
        for _ in range(1, blocks):
            layers_list.append(block(filters))

        return tf.keras.Sequential(layers_list)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

