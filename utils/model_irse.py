import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from collections import namedtuple

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def l2_norm(input_tensor, axis=1):
    norm = tf.norm(input_tensor, ord=2, axis=axis, keepdims=True)
    output = tf.divide(input_tensor, norm)
    return output

class SEModule(layers.Layer):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(channels // reduction, kernel_size=1, padding='valid', use_bias=False)
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(channels, kernel_size=1, padding='valid', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
        
        # Initialize weights
        self.fc1.build((None, None, None, channels))
        tf.keras.initializers.GlorotUniform()(self.fc1.kernel)

    def call(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class BottleneckIR(layers.Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        
        if in_channel == depth:
            self.shortcut_layer = layers.MaxPool2D(pool_size=1, strides=stride)
        else:
            self.shortcut_layer = tf.keras.Sequential([
                layers.Conv2D(depth, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
            
        self.res_layer = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(depth, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.PReLU(shared_axes=[1, 2]),
            layers.Conv2D(depth, kernel_size=3, strides=stride, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

    def call(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class BottleneckIRSE(layers.Layer):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__()
        
        if in_channel == depth:
            self.shortcut_layer = layers.MaxPool2D(pool_size=1, strides=stride)
        else:
            self.shortcut_layer = tf.keras.Sequential([
                layers.Conv2D(depth, kernel_size=1, strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
            
        self.res_layer = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(depth, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.PReLU(shared_axes=[1, 2]),
            layers.Conv2D(depth, kernel_size=3, strides=stride, padding='same', use_bias=False),
            layers.BatchNormalization(),
            SEModule(depth, 16)
        ])

    def call(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

Bottleneck = namedtuple('Block', ['in_channel', 'depth', 'stride'])

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks

class Backbone(Model):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        
        self.input_layer = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.PReLU(shared_axes=[1, 2])
        ])
        
        if input_size[0] == 112:
            self.output_layer = tf.keras.Sequential([
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(512, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.output_layer = tf.keras.Sequential([
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(512, use_bias=False),
                layers.BatchNormalization()
            ])
        
        blocks = get_blocks(num_layers)
        unit_module = BottleneckIRSE if mode == 'ir_se' else BottleneckIR
        
        self.body = tf.keras.Sequential()
        for block in blocks:
            for bottleneck in block:
                self.body.add(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        
        self._initialize_weights()

    def call(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, layers.Conv2D):
                tf.keras.initializers.GlorotUniform()(layer.kernel)
                if layer.use_bias:
                    tf.zeros_initializer()(layer.bias)
            elif isinstance(layer, layers.BatchNormalization):
                tf.ones_initializer()(layer.gamma)
                tf.zeros_initializer()(layer.beta)
            elif isinstance(layer, layers.Dense):
                tf.keras.initializers.GlorotUniform()(layer.kernel)
                if layer.use_bias:
                    tf.zeros_initializer()(layer.bias)

def IR_50(input_size):
    """Constructs a ir-50 model."""
    return Backbone(input_size, 50, 'ir')

def IR_101(input_size):
    """Constructs a ir-101 model."""
    return Backbone(input_size, 100, 'ir')

def IR_152(input_size):
    """Constructs a ir-152 model."""
    return Backbone(input_size, 152, 'ir')

def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    return Backbone(input_size, 50, 'ir_se')

def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    return Backbone(input_size, 100, 'ir_se')

def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    return Backbone(input_size, 152, 'ir_se')

if __name__ == '__main__':
    # Test the model
    input_size = [112, 112]
    model = IR_50(input_size)
    
    # Create a sample input
    x = tf.random.normal((1, 112, 112, 3))
    
    # Forward pass
    output = model(x)
    print(f"Model output shape: {output.shape}")
    
    # Test other variants
    models = [
        IR_101(input_size),
        IR_152(input_size),
        IR_SE_50(input_size),
        IR_SE_101(input_size),
        IR_SE_152(input_size)
    ]
    
    for i, m in enumerate(models):
        output = m(x)
        print(f"Model {i+1} output shape: {output.shape}")
