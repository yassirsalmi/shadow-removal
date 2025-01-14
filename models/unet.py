import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ListModule(layers.Layer):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        self.modules = list(args)
    
    def __getitem__(self, idx):
        if idx >= len(self.modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx = len(self) + idx
        return self.modules[idx]
    
    def __iter__(self):
        return iter(self.modules)
    
    def __len__(self):
        return len(self.modules)

def conv(filters, kernel_size, strides=1, use_bias=True, padding='same'):
    return layers.Conv2D(filters, kernel_size, strides=strides,
                        padding=padding, use_bias=use_bias)

class UNetConv2(layers.Layer):
    def __init__(self, filters, norm_layer=layers.BatchNormalization, use_bias=True, padding='same'):
        super(UNetConv2, self).__init__()
        
        if norm_layer is not None:
            self.conv1 = tf.keras.Sequential([
                conv(filters, 3, use_bias=use_bias, padding=padding),
                norm_layer(),
                layers.ReLU()
            ])
            self.conv2 = tf.keras.Sequential([
                conv(filters, 3, use_bias=use_bias, padding=padding),
                norm_layer(),
                layers.ReLU()
            ])
        else:
            self.conv1 = tf.keras.Sequential([
                conv(filters, 3, use_bias=use_bias, padding=padding),
                layers.ReLU()
            ])
            self.conv2 = tf.keras.Sequential([
                conv(filters, 3, use_bias=use_bias, padding=padding),
                layers.ReLU()
            ])
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        return x

class UNetDown(layers.Layer):
    def __init__(self, filters, norm_layer=layers.BatchNormalization, use_bias=True, padding='same'):
        super(UNetDown, self).__init__()
        self.conv = UNetConv2(filters, norm_layer, use_bias, padding)
        self.pool = layers.MaxPool2D(pool_size=2, strides=2)
    
    def call(self, inputs, training=False):
        x = self.pool(inputs)
        x = self.conv(x, training=training)
        return x

class UNetUp(layers.Layer):
    def __init__(self, filters, upsample_mode='deconv', use_bias=True, padding='same', same_num_filt=False):
        super(UNetUp, self).__init__()
        
        num_filt = filters if same_num_filt else filters * 2
        if upsample_mode == 'deconv':
            self.up = layers.Conv2DTranspose(filters, 4, strides=2, padding='same')
            self.conv = UNetConv2(filters, None, use_bias, padding)
        elif upsample_mode in ['bilinear', 'nearest']:
            self.up = tf.keras.Sequential([
                layers.UpSampling2D(size=2, interpolation=upsample_mode),
                conv(filters, 3, use_bias=use_bias, padding=padding)
            ])
            self.conv = UNetConv2(filters, None, use_bias, padding)
        else:
            raise ValueError(f"Unsupported upsample mode: {upsample_mode}")
    
    def call(self, inputs1, inputs2, training=False):
        x = self.up(inputs1)
        
        # Handle size mismatch
        if tf.shape(inputs2)[1] != tf.shape(x)[1] or tf.shape(inputs2)[2] != tf.shape(x)[2]:
            diff_h = (tf.shape(inputs2)[1] - tf.shape(x)[1]) // 2
            diff_w = (tf.shape(inputs2)[2] - tf.shape(x)[2]) // 2
            inputs2 = inputs2[:, diff_h:diff_h + tf.shape(x)[1], 
                            diff_w:diff_w + tf.shape(x)[2], :]
        
        x = tf.concat([x, inputs2], axis=-1)
        x = self.conv(x, training=training)
        return x

class UNet(Model):
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 feature_scale=4, more_layers=0, concat_x=False,
                 upsample_mode='deconv', padding='same', 
                 norm_layer=layers.BatchNormalization,
                 need_sigmoid=True, use_bias=True):
        super(UNet, self).__init__()
        
        # Force TensorFlow to use CPU
        tf.config.set_visible_devices([], 'GPU')
        
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x
        
        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]
        
        # Initial path
        self.start = UNetConv2(filters[0] if not concat_x else filters[0] - num_input_channels,
                              norm_layer, use_bias, padding)
        
        # Downsampling path
        self.down1 = UNetDown(filters[1] if not concat_x else filters[1] - num_input_channels,
                             norm_layer, use_bias, padding)
        self.down2 = UNetDown(filters[2] if not concat_x else filters[2] - num_input_channels,
                             norm_layer, use_bias, padding)
        self.down3 = UNetDown(filters[3] if not concat_x else filters[3] - num_input_channels,
                             norm_layer, use_bias, padding)
        self.down4 = UNetDown(filters[4] if not concat_x else filters[4] - num_input_channels,
                             norm_layer, use_bias, padding)
        
        # Additional layers
        if self.more_layers > 0:
            self.more_downs = [
                UNetDown(filters[4] if not concat_x else filters[4] - num_input_channels,
                        norm_layer, use_bias, padding) for _ in range(self.more_layers)
            ]
            self.more_ups = [
                UNetUp(filters[4], upsample_mode, use_bias, padding, same_num_filt=True)
                for _ in range(self.more_layers)
            ]
            
            self.more_downs = ListModule(*self.more_downs)
            self.more_ups = ListModule(*self.more_ups)
        
        # Upsampling path
        self.up4 = UNetUp(filters[3], upsample_mode, use_bias, padding)
        self.up3 = UNetUp(filters[2], upsample_mode, use_bias, padding)
        self.up2 = UNetUp(filters[1], upsample_mode, use_bias, padding)
        self.up1 = UNetUp(filters[0], upsample_mode, use_bias, padding)
        
        # Final convolution
        self.final = conv(num_output_channels, 1, use_bias=use_bias, padding=padding)
        self.need_sigmoid = need_sigmoid
    
    def call(self, inputs, training=False):
        # Downsample for skip connections
        downs = [inputs]
        down = layers.AveragePooling2D(pool_size=2, strides=2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))
        
        # Initial block
        x = self.start(inputs, training=training)
        if self.concat_x:
            x = tf.concat([x, downs[0]], axis=-1)
        
        # Encoder path
        down1 = self.down1(x, training=training)
        if self.concat_x:
            down1 = tf.concat([down1, downs[1]], axis=-1)
        
        down2 = self.down2(down1, training=training)
        if self.concat_x:
            down2 = tf.concat([down2, downs[2]], axis=-1)
        
        down3 = self.down3(down2, training=training)
        if self.concat_x:
            down3 = tf.concat([down3, downs[3]], axis=-1)
        
        down4 = self.down4(down3, training=training)
        if self.concat_x:
            down4 = tf.concat([down4, downs[4]], axis=-1)
        
        # Additional layers if any
        if self.more_layers > 0:
            prevs = [down4]
            for i, d in enumerate(self.more_downs):
                out = d(prevs[-1], training=training)
                if self.concat_x:
                    out = tf.concat([out, downs[i + 5]], axis=-1)
                prevs.append(out)
            
            up_ = self.more_ups[-1](prevs[-1], prevs[-2], training=training)
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more_layers - idx - 2]
                up_ = l(up_, prevs[self.more_layers - idx - 2], training=training)
        else:
            up_ = down4
        
        # Decoder path
        up4 = self.up4(up_, down3, training=training)
        up3 = self.up3(up4, down2, training=training)
        up2 = self.up2(up3, down1, training=training)
        up1 = self.up1(up2, x, training=training)
        
        # Final convolution
        x = self.final(up1)
        if self.need_sigmoid:
            x = tf.sigmoid(x)
        
        return x

if __name__ == "__main__":
    # Test the model
    model = UNet()
    test_input = tf.random.normal((1, 256, 256, 3))
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
