#!/usr/bin/python
# -*- encoding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBNReLU(Model):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding="same"):
        super(ConvBNReLU, self).__init__()
        self.conv = layers.Conv2D(out_chan, kernel_size=ks, strides=stride, padding=padding, use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class BiSeNetOutput(Model):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = layers.Conv2D(n_classes, kernel_size=1, use_bias=False)

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(Model):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = layers.Conv2D(out_chan, kernel_size=1, use_bias=False)
        self.bn_atten = layers.BatchNormalization()
        self.sigmoid_atten = layers.Activation("sigmoid")

    def call(self, x, training=False):
        feat = self.conv(x, training=training)
        atten = tf.reduce_mean(feat, axis=[1, 2], keepdims=True)  # Global Average Pooling
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten, training=training)
        atten = self.sigmoid_atten(atten)
        out = feat * atten
        return out


class ContextPath(Model):
    def __init__(self, resnet):
        super(ContextPath, self).__init__()
        self.resnet = resnet
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding="valid")

    def call(self, x, training=False):
        feat8, feat16, feat32 = self.resnet(x, training=training)

        avg = tf.reduce_mean(feat32, axis=[1, 2], keepdims=True)
        avg = self.conv_avg(avg, training=training)
        avg_up = tf.image.resize(avg, tf.shape(feat32)[1:3])

        feat32_arm = self.arm32(feat32, training=training)
        feat32_sum = feat32_arm + avg_up
        feat32_up = tf.image.resize(feat32_sum, tf.shape(feat16)[1:3])
        feat32_up = self.conv_head32(feat32_up, training=training)

        feat16_arm = self.arm16(feat16, training=training)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = tf.image.resize(feat16_sum, tf.shape(feat8)[1:3])
        feat16_up = self.conv_head16(feat16_up, training=training)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(Model):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, padding="valid")
        self.conv1 = layers.Conv2D(out_chan // 4, kernel_size=1, use_bias=False)
        self.conv2 = layers.Conv2D(out_chan, kernel_size=1, use_bias=False)
        self.relu = layers.ReLU()
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, fsp, fcp, training=False):
        fcat = tf.concat([fsp, fcp], axis=-1)
        feat = self.convblk(fcat, training=training)

        atten = tf.reduce_mean(feat, axis=[1, 2], keepdims=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = feat * atten
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(Model):
    def __init__(self, n_classes):
        super(BiSeNet, self).__init__()
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights=None)
        self.cp = ContextPath(self.resnet)
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def call(self, x, training=False):
        feat_res8, feat_cp8, feat_cp16 = self.cp(x, training=training)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8, training=training)

        feat_out = self.conv_out(feat_fuse, training=training)
        feat_out16 = self.conv_out16(feat_cp8, training=training)
        feat_out32 = self.conv_out32(feat_cp16, training=training)

        feat_out = tf.image.resize(feat_out, tf.shape(x)[1:3])
        feat_out16 = tf.image.resize(feat_out16, tf.shape(x)[1:3])
        feat_out32 = tf.image.resize(feat_out32, tf.shape(x)[1:3])

        return feat_out, feat_out16, feat_out32


if __name__ == "__main__":
    net = BiSeNet(n_classes=19)
    net.build(input_shape=(None, 640, 480, 3))
    net.summary()

    in_tensor = tf.random.normal([16, 640, 480, 3])
    out, out16, out32 = net(in_tensor)
