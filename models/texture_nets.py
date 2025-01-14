import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class Concat(layers.Layer):
    def __init__(self, axis=-1):
        super(Concat, self).__init__()
        self.axis = axis
    
    def call(self, inputs):
        return tf.concat(inputs, axis=self.axis)

class GenNoise(layers.Layer):
    def __init__(self, channels):
        super(GenNoise, self).__init__()
        self.channels = channels
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        noise = tf.random.normal([batch_size, height, width, self.channels])
        return inputs + noise

def conv_block(filters, kernel_size=3, stride=1, use_bias=True, padding='same'):
    if padding == 'reflection':
        def _conv_block(x):
            x = layers.Lambda(lambda x: tf.pad(x, [[0,0], [kernel_size//2,kernel_size//2], 
                                                  [kernel_size//2,kernel_size//2], [0,0]], 
                                             mode='REFLECT'))(x)
            x = layers.Conv2D(filters, kernel_size, strides=stride, 
                            padding='valid', use_bias=use_bias)(x)
            return x
        return _conv_block
    else:
        return layers.Conv2D(filters, kernel_size, strides=stride, 
                           padding=padding, use_bias=use_bias)

class TextureNet(Model):
    def __init__(self, inp=3, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False,
                 pad='same', need_sigmoid=False, conv_num=8, upsample_mode='nearest'):
        super(TextureNet, self).__init__()
        
        self.model_layers = []
        self.ratios = ratios
        self.fill_noise = fill_noise
        self.pad = pad
        self.need_sigmoid = need_sigmoid
        self.conv_num = conv_num
        self.upsample_mode = upsample_mode
        
        # Force TensorFlow to use CPU
        tf.config.set_visible_devices([], 'GPU')
        
        self.build_network(inp)
    
    def build_network(self, inp):
        cur = None
        
        for i, ratio in enumerate(self.ratios):
            j = i + 1
            
            # Create sequential block
            seq = []
            
            # Average pooling
            seq.append(layers.AveragePooling2D(pool_size=ratio, strides=ratio))
            
            if self.fill_noise:
                seq.append(GenNoise(inp))
            
            # First conv block
            seq.append(conv_block(self.conv_num, 3, padding=self.pad))
            seq.append(layers.BatchNormalization())
            seq.append(layers.LeakyReLU(alpha=0.2))
            
            # Second conv block
            seq.append(conv_block(self.conv_num, 3, padding=self.pad))
            seq.append(layers.BatchNormalization())
            seq.append(layers.LeakyReLU(alpha=0.2))
            
            # 1x1 conv block
            seq.append(conv_block(self.conv_num, 1, padding=self.pad))
            seq.append(layers.BatchNormalization())
            seq.append(layers.LeakyReLU(alpha=0.2))
            
            if i == 0:
                seq.append(layers.UpSampling2D(size=2, interpolation=self.upsample_mode))
                cur = tf.keras.Sequential(seq)
            else:
                cur_temp = cur
                
                # Batch norm before merging
                seq.append(layers.BatchNormalization())
                cur_temp = tf.keras.Sequential([cur_temp, layers.BatchNormalization()])
                
                # Concatenate previous and current paths
                merged = Concat()([cur_temp.output, tf.keras.Sequential(seq).output])
                
                # New conv blocks after concatenation
                x = conv_block(self.conv_num * j, 3, padding=self.pad)(merged)
                x = layers.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.2)(x)
                
                x = conv_block(self.conv_num * j, 3, padding=self.pad)(x)
                x = layers.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.2)(x)
                
                x = conv_block(self.conv_num * j, 1, padding=self.pad)(x)
                x = layers.BatchNormalization()(x)
                x = layers.LeakyReLU(alpha=0.2)(x)
                
                if i == len(self.ratios) - 1:
                    x = conv_block(3, 1, padding=self.pad)(x)
                else:
                    x = layers.UpSampling2D(size=2, interpolation=self.upsample_mode)(x)
                
                cur = Model(inputs=[cur_temp.input, tf.keras.Sequential(seq).input], outputs=x)
        
        self.model = cur
        if self.need_sigmoid:
            self.model = tf.keras.Sequential([self.model, layers.Activation('sigmoid')])
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

def get_texture_nets(inp=3, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False,
                    pad='same', need_sigmoid=False, conv_num=8, upsample_mode='nearest'):
    """Creates a texture network model with specified parameters."""
    return TextureNet(inp, ratios, fill_noise, pad, need_sigmoid, conv_num, upsample_mode)

if __name__ == "__main__":
    # Test the model
    model = get_texture_nets()
    test_input = tf.random.normal((1, 256, 256, 3))
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
