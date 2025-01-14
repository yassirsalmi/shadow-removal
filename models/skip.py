import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Concat(layers.Layer):
    def __init__(self, axis=-1):
        super(Concat, self).__init__()
        self.axis = axis
    
    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)

def get_activation(act_fun):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return layers.LeakyReLU(alpha=0.2)
        elif act_fun == 'ELU':
            return layers.ELU()
        elif act_fun == 'none':
            return layers.Lambda(lambda x: x)
        else:
            raise ValueError(f'Unknown activation function: {act_fun}')
    return act_fun

def conv(filters, kernel_size, strides=1, padding='same', use_bias=True):
    return layers.Conv2D(filters, kernel_size, strides=strides,
                        padding=padding, use_bias=use_bias)

def bn(filters):
    return layers.BatchNormalization()

def create_skip_model(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3,
        filter_skip_size=1, need_sigmoid=True, need_bias=True, pad='same',
        upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections in TensorFlow.

    Arguments:
        act_fun: Either string 'LeakyReLU|ELU|none' or a callable
        pad (string): same|valid (default: 'same')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max' (default: 'stride')
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    
    n_scales = len(num_channels_down)
    
    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    class SkipModel(Model):
        def __init__(self):
            super(SkipModel, self).__init__()
            self.encoder_blocks = []
            self.decoder_blocks = []
            self.skip_blocks = []
            
            activation = get_activation(act_fun)
            
            # Build encoder and skip connections
            input_depth = num_input_channels
            for i in range(n_scales):
                encoder_block = tf.keras.Sequential([
                    conv(num_channels_down[i], filter_size_down[i], strides=2, use_bias=need_bias),
                    bn(num_channels_down[i]),
                    activation,
                    conv(num_channels_down[i], filter_size_down[i], use_bias=need_bias),
                    bn(num_channels_down[i]),
                    activation
                ])
                
                skip_block = tf.keras.Sequential([]) if num_channels_skip[i] == 0 else tf.keras.Sequential([
                    conv(num_channels_skip[i], filter_skip_size, use_bias=need_bias),
                    bn(num_channels_skip[i]),
                    activation
                ])
                
                self.encoder_blocks.append(encoder_block)
                self.skip_blocks.append(skip_block)
                input_depth = num_channels_down[i]
            
            # Build decoder
            for i in range(n_scales):
                decoder_block = tf.keras.Sequential()
                
                # Upsampling
                if upsample_mode[i] == 'nearest':
                    decoder_block.add(layers.UpSampling2D(size=2, interpolation='nearest'))
                else:
                    decoder_block.add(layers.UpSampling2D(size=2, interpolation='bilinear'))
                
                # Convolution after concatenation
                in_channels = num_channels_skip[i] + (num_channels_up[i + 1] if i < n_scales - 1 else num_channels_down[i])
                decoder_block.add(conv(num_channels_up[i], filter_size_up[i], use_bias=need_bias))
                decoder_block.add(bn(num_channels_up[i]))
                decoder_block.add(activation)
                
                if need1x1_up:
                    decoder_block.add(conv(num_channels_up[i], num_channels_up[i], 1, use_bias=need_bias))
                    decoder_block.add(bn(num_channels_up[i]))
                    decoder_block.add(activation)
                
                self.decoder_blocks.append(decoder_block)
            
            # Final convolution
            self.final_conv = conv(num_output_channels, 1, use_bias=need_bias)
            self.need_sigmoid = need_sigmoid
        
        def call(self, inputs, training=False):
            x = inputs
            encoder_features = []
            
            # Encoder path with skip connections
            for encoder, skip in zip(self.encoder_blocks, self.skip_blocks):
                if skip.layers:  # If skip connection exists
                    skip_features = skip(x, training=training)
                    encoder_features.append(skip_features)
                
                x = encoder(x, training=training)
            
            # Decoder path
            for i, decoder in enumerate(self.decoder_blocks):
                if i < len(encoder_features):  # If there are skip features to concatenate
                    x = tf.concat([x, encoder_features[-(i+1)]], axis=-1)
                x = decoder(x, training=training)
            
            x = self.final_conv(x)
            if self.need_sigmoid:
                x = tf.sigmoid(x)
            
            return x
    
    # Force TensorFlow to use CPU
    tf.config.set_visible_devices([], 'GPU')
    
    return SkipModel()

if __name__ == "__main__":
    # Test the model
    model = create_skip_model()
    test_input = tf.random.normal((1, 256, 256, 2))
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
