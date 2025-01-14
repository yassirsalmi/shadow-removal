import tensorflow as tf
import numpy as np


class Concat(tf.keras.layers.Layer):
    def __init__(self, dim, *args):
        """
        Concatenates multiple layers along the specified dimension.
        """
        super(Concat, self).__init__()
        self.dim = dim
        self.modules = list(args)

    def call(self, inputs, training=False):
        outputs = [module(inputs, training=training) for module in self.modules]

        # Match spatial dimensions if needed
        heights = [x.shape[1] for x in outputs]
        widths = [x.shape[2] for x in outputs]

        if np.all(np.array(heights) == min(heights)) and np.all(np.array(widths) == min(widths)):
            aligned_outputs = outputs
        else:
            target_height = min(heights)
            target_width = min(widths)

            aligned_outputs = []
            for output in outputs:
                diff_height = (output.shape[1] - target_height) // 2
                diff_width = (output.shape[2] - target_width) // 2
                aligned_outputs.append(
                    tf.image.crop_to_bounding_box(output, diff_height, diff_width, target_height, target_width)
                )

        return tf.concat(aligned_outputs, axis=self.dim)


class GenNoise(tf.keras.layers.Layer):
    def __init__(self, dim2):
        """
        Generates random noise with the specified number of channels.
        """
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def call(self, inputs):
        shape = list(inputs.shape)
        shape[1] = self.dim2  # Set the desired number of channels
        noise = tf.random.normal(shape)
        return noise


class Swish(tf.keras.layers.Layer):
    """
    Swish activation function.
    Reference: https://arxiv.org/abs/1710.05941
    """
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


def act(act_fun='LeakyReLU'):
    """
    Returns an activation function layer based on the specified string or layer.
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return tf.keras.layers.LeakyReLU(alpha=0.2)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return tf.keras.layers.ELU()
        elif act_fun == 'none':
            return tf.keras.layers.Activation('linear')
        else:
            raise ValueError(f"Unsupported activation function: {act_fun}")
    else:
        return act_fun


def bn(num_features):
    """
    Batch normalization layer for 2D inputs.
    """
    return tf.keras.layers.BatchNormalization()


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """
    Creates a convolutional layer with optional downsampling and padding.
    """
    layers = []

    # Padding
    to_pad = (kernel_size - 1) // 2
    if pad == 'reflection':
        layers.append(tf.keras.layers.ReflectionPadding2D(padding=to_pad))
        to_pad = 0  # No further padding needed with ReflectionPadding2D

    # Convolution
    layers.append(tf.keras.layers.Conv2D(
        out_f, 
        kernel_size=kernel_size, 
        strides=(1 if downsample_mode != 'stride' else stride), 
        padding=('valid' if pad == 'reflection' else 'same'), 
        use_bias=bias
    ))

    # Downsampler
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            layers.append(tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride))
        elif downsample_mode == 'max':
            layers.append(tf.keras.layers.MaxPooling2D(pool_size=stride, strides=stride))
        elif downsample_mode in ['lanczos2', 'lanczos3']:
            raise NotImplementedError("Lanczos downsampling is not directly supported in TensorFlow.")
        else:
            raise ValueError(f"Unsupported downsampling mode: {downsample_mode}")

    return tf.keras.Sequential(layers)
