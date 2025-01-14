import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, 
    ResNet101, 
    ResNet152, 
    VGG16, 
    VGG19, 
    MobileNetV2,
    DenseNet121
)
from collections import namedtuple

class SqueezeNet(tf.keras.Model):
    def __init__(self, requires_grad=False, pretrained=True):
        super(SqueezeNet, self).__init__()
        # SqueezeNet is not directly available in Keras; we use a smaller MobileNetV2 as a substitute
        base_model = tf.keras.applications.MobileNetV2(weights="imagenet" if pretrained else None, include_top=False)
        layers = base_model.layers
        self.slice1 = tf.keras.Sequential(layers[:3])   # Slicing the model layers
        self.slice2 = tf.keras.Sequential(layers[3:6])
        self.slice3 = tf.keras.Sequential(layers[6:9])
        self.slice4 = tf.keras.Sequential(layers[9:12])
        self.slice5 = tf.keras.Sequential(layers[12:15])
        self.slice6 = tf.keras.Sequential(layers[15:18])
        self.slice7 = tf.keras.Sequential(layers[18:])
        self.N_slices = 7

        if not requires_grad:
            for layer in base_model.layers:
                layer.trainable = False

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        h_relu6 = self.slice6(h_relu5)
        h_relu7 = self.slice7(h_relu6)

        SqueezeOutputs = namedtuple("SqueezeOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"])
        return SqueezeOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)


class AlexNet(tf.keras.Model):
    def __init__(self, requires_grad=False, pretrained=True):
        # Using VGG16 as AlexNet isn't natively available in TensorFlow
        super(AlexNet, self).__init__()
        base_model = tf.keras.applications.VGG16(weights="imagenet" if pretrained else None, include_top=False)
        layers = base_model.layers
        self.slice1 = tf.keras.Sequential(layers[:3])
        self.slice2 = tf.keras.Sequential(layers[3:6])
        self.slice3 = tf.keras.Sequential(layers[6:9])
        self.slice4 = tf.keras.Sequential(layers[9:12])
        self.slice5 = tf.keras.Sequential(layers[12:])
        self.N_slices = 5

        if not requires_grad:
            for layer in base_model.layers:
                layer.trainable = False

    def call(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)

        AlexNetOutputs = namedtuple("AlexNetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        return AlexNetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class VGG16Model(tf.keras.Model):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16Model, self).__init__()
        base_model = VGG16(weights="imagenet" if pretrained else None, include_top=False)
        layers = base_model.layers
        self.slice1 = tf.keras.Sequential(layers[:4])
        self.slice2 = tf.keras.Sequential(layers[4:9])
        self.slice3 = tf.keras.Sequential(layers[9:16])
        self.slice4 = tf.keras.Sequential(layers[16:23])
        self.slice5 = tf.keras.Sequential(layers[23:])
        self.N_slices = 5

        if not requires_grad:
            for layer in base_model.layers:
                layer.trainable = False

    def call(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)

        VGGOutputs = namedtuple("VGGOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class ResNet(tf.keras.Model):
    def __init__(self, num=50, requires_grad=False, pretrained=True):
        super(ResNet, self).__init__()
        if num == 50:
            base_model = ResNet50(weights="imagenet" if pretrained else None, include_top=False)
        elif num == 101:
            base_model = ResNet101(weights="imagenet" if pretrained else None, include_top=False)
        elif num == 152:
            base_model = ResNet152(weights="imagenet" if pretrained else None, include_top=False)
        else:
            raise ValueError("Unsupported ResNet version!")

        self.conv1 = base_model.get_layer("conv1_conv")
        self.bn1 = base_model.get_layer("conv1_bn")
        self.relu = base_model.get_layer("conv1_relu")
        self.maxpool = base_model.get_layer("pool1_pool")
        self.layer1 = base_model.get_layer("conv2_block1_1_conv")
        self.layer2 = base_model.get_layer("conv2_block2_1_conv")
        self.layer3 = base_model.get_layer("conv3_block1_1_conv")
        self.layer4 = base_model.get_layer("conv4_block1_1_conv")
        self.N_slices = 5

        if not requires_grad:
            for layer in base_model.layers:
                layer.trainable = False

    def call(self, x):
        h_relu1 = self.relu(self.bn1(self.conv1(x)))
        h_conv2 = self.maxpool(h_relu1)
        h_conv3 = self.layer1(h_conv2)
        h_conv4 = self.layer2(h_conv3)
        h_conv5 = self.layer3(h_conv4)

        ResNetOutputs = namedtuple("ResNetOutputs", ["relu1", "conv2", "conv3", "conv4", "conv5"])
        return ResNetOutputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
