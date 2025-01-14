import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.config.set_visible_devices([], 'GPU')

class PixelNorm(layers.Layer):
    def call(self, inputs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + 1e-8)

def make_kernel(k):
    k = tf.convert_to_tensor(k, dtype=tf.float32)
    if len(k.shape) == 1:
        k = tf.expand_dims(k, 0) * tf.expand_dims(k, 1)
    k = k / tf.reduce_sum(k)
    return k

class Upsample(layers.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.kernel = tf.Variable(kernel, trainable=False)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        self.pad = [[0, 0], [pad0, pad1], [pad0, pad1], [0, 0]]

    def call(self, inputs):
        shape = tf.shape(inputs)
        h = shape[1] * self.factor
        w = shape[2] * self.factor
        x = tf.image.resize(inputs, [h, w], method='nearest')
        kernel = tf.reshape(self.kernel, [1, -1, 1, 1])
        x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return x

class Downsample(layers.Layer):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.kernel = tf.Variable(kernel, trainable=False)
        p = kernel.shape[0] - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        self.pad = [[0, 0], [pad0, pad1], [pad0, pad1], [0, 0]]

    def call(self, inputs):
        x = tf.pad(inputs, self.pad)
        kernel = tf.reshape(self.kernel, [1, -1, 1, 1])
        x = tf.nn.conv2d(x, kernel, strides=[1, self.factor, self.factor, 1], padding='VALID')
        return x

class EqualConv2D(layers.Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding='same', use_bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.scale = 1 / math.sqrt(in_channels * self.kernel_size ** 2)
        
        self.weight = self.add_weight(
            'weight',
            shape=[self.kernel_size, self.kernel_size, in_channels, self.out_channels],
            initializer=tf.random_normal_initializer(0, 1),
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.out_channels],
                initializer=tf.zeros_initializer(),
                trainable=True
            )
    
    def call(self, inputs):
        weight = self.weight * self.scale
        x = tf.nn.conv2d(inputs, weight, strides=[1, self.stride, self.stride, 1], padding=self.padding.upper())
        if self.use_bias:
            x = x + self.bias
        return x

class EqualLinear(layers.Layer):
    def __init__(self, out_dim, use_bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.lr_mul = lr_mul
        self.activation = activation

    def build(self, input_shape):
        in_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape[-1]
        
        self.scale = (1 / math.sqrt(in_dim)) * self.lr_mul
        
        self.weight = self.add_weight(
            name='weight',
            shape=[in_dim, self.out_dim],
            initializer=tf.random_normal_initializer(0, 1/self.lr_mul),
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.out_dim],
                initializer=tf.constant_initializer(self.bias_init),
                trainable=True
            )
        
        super().build(input_shape) 

    def call(self, inputs):
        x = tf.matmul(inputs, self.weight * self.scale)
        if self.use_bias:
            x = x + self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

class ModulatedConv2D(layers.Layer):
    def __init__(
        self,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        
        if upsample:
            self.blur = Upsample(blur_kernel)
        elif downsample:
            self.blur = Downsample(blur_kernel)

        self.modulation = EqualLinear(style_dim, use_bias=True, bias_init=1)
        
    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.in_channel = in_channel
        
        self.scale = 1 / math.sqrt(in_channel * self.kernel_size ** 2)
        self.padding = self.kernel_size // 2
        
        self.weight = self.add_weight(
            'weight',
            shape=[1, self.out_channel, in_channel, self.kernel_size, self.kernel_size],
            initializer=tf.random_normal_initializer(0, 1),
            trainable=True
        )

    def call(self, inputs, style):
        batch = tf.shape(inputs)[0]
        
        # Modulation
        style = self.modulation(style)
        style = tf.reshape(style, [batch, 1, self.in_channel, 1, 1])
        weight = self.scale * self.weight * style
        
        # Demodulation
        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[2, 3, 4]) + 1e-8)
            weight = weight * tf.reshape(demod, [batch, self.out_channel, 1, 1, 1])
        
        weight = tf.reshape(
            weight, [batch * self.out_channel, self.in_channel, self.kernel_size, self.kernel_size]
        )
        
        if self.upsample:
            inputs = tf.reshape(inputs, [1, batch * self.in_channel, tf.shape(inputs)[1], tf.shape(inputs)[2]])
            weight = tf.reshape(weight, [batch, self.out_channel, self.in_channel, self.kernel_size, self.kernel_size])
            weight = tf.transpose(weight, [0, 2, 1, 3, 4])
            weight = tf.reshape(weight, [batch * self.in_channel, self.out_channel, self.kernel_size, self.kernel_size])
            
            x = tf.nn.conv2d_transpose(
                inputs,
                weight,
                output_shape=[1, batch * self.out_channel, tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2],
                strides=[1, 2, 2, 1],
                padding='SAME'
            )
            x = tf.reshape(x, [batch, self.out_channel, tf.shape(x)[2], tf.shape(x)[3]])
            x = self.blur(x)
            
        elif self.downsample:
            inputs = self.blur(inputs)
            inputs = tf.reshape(inputs, [1, batch * self.in_channel, tf.shape(inputs)[1], tf.shape(inputs)[2]])
            x = tf.nn.conv2d(
                inputs,
                weight,
                strides=[1, 2, 2, 1],
                padding='SAME'
            )
            x = tf.reshape(x, [batch, self.out_channel, tf.shape(x)[1], tf.shape(x)[2]])
            
        else:
            inputs = tf.reshape(inputs, [1, batch * self.in_channel, tf.shape(inputs)[1], tf.shape(inputs)[2]])
            x = tf.nn.conv2d(
                inputs,
                weight,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            x = tf.reshape(x, [batch, self.out_channel, tf.shape(x)[1], tf.shape(x)[2]])
            
        return x

class NoiseInjection(layers.Layer):
    def build(self, input_shape):
        self.weight = self.add_weight(
            'weight',
            shape=[1],
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        
    def call(self, inputs, noise=None):
        if noise is None:
            batch, _, height, width = tf.shape(inputs)
            noise = tf.random.normal([batch, 1, height, width])
        return inputs + self.weight * noise

class ConstantInput(layers.Layer):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input_tensor = tf.Variable(
            tf.random.normal([1, channel, size, size]),
            trainable=True
        )
        
    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        return tf.tile(self.input_tensor, [batch, 1, 1, 1])

class StyledConv(layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        
        self.conv = ModulatedConv2D(
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        
        self.noise = NoiseInjection()
        self.activate = layers.LeakyReLU(0.2)
        
    def call(self, inputs, style, noise=None):
        x = self.conv(inputs, style)
        x = self.noise(x, noise=noise)
        x = self.activate(x)
        return x

class Generator(keras.Model):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        # Style network
        self.style = tf.keras.Sequential([PixelNorm()])
        for _ in range(n_mlp):
            self.style.add(
                EqualLinear(style_dim, lr_mul=lr_mlp, activation='fused_lrelu')
            )
            
        self.log_size = int(np.log2(size))

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input_tensor = ConstantInput(self.channels[4])

        self.resnet_layers = []

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.resnet_layers.append(
                layers.Conv2D(out_channel, 3, strides=1, padding="same", use_bias=False)
            )
            self.resnet_layers.append(
                layers.BatchNormalization()
            )
            self.resnet_layers.append(
                layers.ReLU()
            )

            self.resnet_layers.append(
                layers.Add()
            )

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.to_rgb = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=1,
            padding='same',
            name='to_rgb'
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, style=None, noise=None):
        #print(f"[DEBUG] Call received inputs type: {type(inputs)}")
        #print(f"[DEBUG] Inputs value: {inputs}")
        
        # If inputs is a list, unpack it
        if isinstance(inputs, list):
            #print(f"[DEBUG] Input is list with length: {len(inputs)}")
            latent_input, style = inputs
            #print(f"[DEBUG] Unpacked latent_input shape: {latent_input.shape}")
            #print(f"[DEBUG] Unpacked style shape: {style.shape}")
        else:
            #print(f"[DEBUG] Input is not a list, shape: {inputs.shape}")
            latent_input = inputs

        # Normalize and process style
        if style is not None:
            #print(f"[DEBUG] Processing style input")
            style = self.style(style)
            #print(f"[DEBUG] Processed style shape: {style.shape}")

        #print(f"[DEBUG] Applying input tensor")
        x = self.input_tensor(latent_input)
        #print(f"[DEBUG] After input tensor shape: {x.shape}")
        
        # Store skip connections with channel matching
        skip_connections = {}
        skip_conv_layers = {}  # Add conv layers for channel matching

        for i, layer in enumerate(self.resnet_layers):
            #print(f"[DEBUG] Processing layer {i}, type: {type(layer)}")
            #print(f"[DEBUG] Input shape to layer {i}: {x.shape}")
            
            try:
                if isinstance(layer, tf.keras.layers.Add):
                    # Get corresponding skip connection
                    skip_x = skip_connections.get(i, None)
                    
                    if skip_x is None:
                        skip_connections[i] = x
                        x = layer([x, x])
                    else:
                        # Match channels if needed
                        if skip_x.shape[-1] != x.shape[-1]:
                            if i not in skip_conv_layers:
                                skip_conv_layers[i] = tf.keras.layers.Conv2D(
                                    x.shape[-1], 
                                    kernel_size=1, 
                                    padding='same'
                                )
                            skip_x = skip_conv_layers[i](skip_x)
                        x = layer([x, skip_x])
                    
                    skip_connections[i] = x
                else:
                    x = layer(x)
                
                #print(f"[DEBUG] Layer {i} output shape: {x.shape}")
                
            except Exception as e:
                #print(f"[ERROR] Failed at layer {i}")
                #print(f"[ERROR] Layer type: {type(layer)}")
                #print(f"[ERROR] Input tensor shape: {x.shape}")
                #print(f"[ERROR] Exception: {str(e)}")
                raise
            
        # Use stored RGB conv layer
        x = self.to_rgb(x)
        #print(f"[DEBUG] Final output shape: {x.shape}")
    
        return x

class Discriminator(keras.Model):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        
        # Initial convolution
        self.conv1 = EqualConv2D(channels[size], 1)
        
        # Progressive discriminator
        self.convs = []
        in_channel = channels[size]
        
        log_size = int(math.log(size, 2))
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            
            self.convs.append(
                EqualConv2D(out_channel, 3, stride=2)
            )
            
            in_channel = out_channel
            
        self.final_conv = EqualConv2D(channels[4], 3)
        self.final_linear = tf.keras.Sequential([
            layers.Flatten(),
            EqualLinear(channels[4], activation='fused_lrelu'),
            EqualLinear(1)
        ])
        
    def call(self, inputs):
        out = self.conv1(inputs)
        
        for conv in self.convs:
            out = conv(out)
            out = tf.nn.leaky_relu(out, 0.2)
            
        out = self.final_conv(out)
        out = tf.nn.leaky_relu(out, 0.2)
        out = self.final_linear(out)
        
        return out
