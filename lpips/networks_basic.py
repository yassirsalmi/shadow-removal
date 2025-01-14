import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Spatial average equivalent
def spatial_average(tensor, keepdims=True):
    return tf.reduce_mean(tensor, axis=[1, 2], keepdims=keepdims)

# Upsample function equivalent
def upsample(tensor, out_h=64):
    in_h = tf.shape(tensor)[1]
    scale_factor = out_h / in_h
    return tf.image.resize(tensor, size=(out_h, out_h), method='bilinear')

# Scaling Layer
class ScalingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = tf.constant([[-0.030, -0.088, -0.188]], dtype=tf.float32)
        self.scale = tf.constant([[0.458, 0.448, 0.450]], dtype=tf.float32)

    def call(self, inputs):
        shift = tf.reshape(self.shift, [1, 1, 1, 3])
        scale = tf.reshape(self.scale, [1, 1, 1, 3])
        return (inputs - shift) / scale

# NetLinLayer: A single 1x1 Conv Layer
class NetLinLayer(tf.keras.layers.Layer):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        self.model = tf.keras.Sequential()
        if use_dropout:
            self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Conv2D(chn_out, kernel_size=1, strides=1, use_bias=False))

    def call(self, inputs):
        return self.model(inputs)

# Dist2LogitLayer: Combines distances into a logit output
class Dist2LogitLayer(tf.keras.layers.Layer):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        self.layers = [
            layers.Conv2D(chn_mid, kernel_size=1, strides=1, activation='leaky_relu'),
            layers.Conv2D(chn_mid, kernel_size=1, strides=1, activation='leaky_relu'),
            layers.Conv2D(1, kernel_size=1, strides=1, activation='sigmoid' if use_sigmoid else None),
        ]

    def call(self, d0, d1, eps=0.1):
        concat = tf.concat([d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)], axis=-1)
        for layer in self.layers:
            concat = layer(concat)
        return concat

# PNetLin Class
class PNetLin(Model):
    def __init__(self, pnet_type='vgg', pnet_rand=False, pnet_tune=False, use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.spatial = spatial
        self.lpips = lpips
        self.version = version

        # Placeholder for feature extractor; use VGG/AlexNet or others
        if pnet_type in ['vgg', 'vgg16']:
            self.feature_extractor = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
            self.chns = [64, 128, 256, 512, 512]
        elif pnet_type == 'alex':
            self.feature_extractor = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')
            self.chns = [64, 192, 384, 256, 256]
        else:
            raise NotImplementedError(f"{pnet_type} not supported.")

        # Linear layers for each feature level
        self.lins = [NetLinLayer(ch, use_dropout=use_dropout) for ch in self.chns]

    def call(self, in0, in1, ret_per_layer=False):
        # Apply scaling
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)

        # Extract features
        feats0 = self.feature_extractor(in0_input)
        feats1 = self.feature_extractor(in1_input)

        # Compute differences
        diffs = [(f0 - f1) ** 2 for f0, f1 in zip(feats0, feats1)]

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[i](diffs[i]), out_h=tf.shape(in0)[1]) for i in range(len(diffs))]
            else:
                res = [spatial_average(self.lins[i](diffs[i]), keepdims=True) for i in range(len(diffs))]
        else:
            if self.spatial:
                res = [upsample(tf.reduce_sum(diffs[i], axis=-1, keepdims=True), out_h=tf.shape(in0)[1]) for i in range(len(diffs))]
            else:
                res = [spatial_average(tf.reduce_sum(diffs[i], axis=-1, keepdims=True), keepdims=True) for i in range(len(diffs))]

        # Combine results
        val = res[0]
        for r in res[1:]:
            val += r

        return (val, res) if ret_per_layer else val
