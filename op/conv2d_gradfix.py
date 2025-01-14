import contextlib
import tensorflow as tf
import warnings

enabled = True
weight_gradients_disabled = False

@contextlib.contextmanager
def no_weight_gradients():
    """Context manager to temporarily disable weight gradients."""
    global weight_gradients_disabled
    old = weight_gradients_disabled
    weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old

def ensure_tuple(xs, ndim):
    """Convert input to tuple with specified dimension."""
    xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,) * ndim
    return xs

class Conv2DGradFix(tf.keras.layers.Layer):
    def __init__(self, transpose, filters, kernel_size, strides=1, padding='VALID',
                 output_padding=None, dilation_rate=1, groups=1, use_bias=True):
        super(Conv2DGradFix, self).__init__()
        
        # Force TensorFlow to use CPU
        tf.config.set_visible_devices([], 'GPU')
        
        self.transpose = transpose
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.output_padding = output_padding
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (tuple, list)) else (dilation_rate, dilation_rate)
        self.groups = groups
        self.use_bias = use_bias
        
        if not transpose:
            self.conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=padding,
                dilation_rate=self.dilation_rate,
                groups=groups,
                use_bias=use_bias
            )
        else:
            self.conv = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=padding,
                output_padding=output_padding,
                dilation_rate=self.dilation_rate,
                groups=groups,
                use_bias=use_bias
            )

    def build(self, input_shape):
        super(Conv2DGradFix, self).build(input_shape)
        
    def call(self, inputs, training=False):
        return self.conv(inputs)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

@tf.custom_gradient
def conv2d_gradfix(inputs, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    """Custom convolution operation with gradient fixes."""
    
    # Convert inputs to TensorFlow format
    strides = ensure_tuple(stride, 2)
    dilations = ensure_tuple(dilation, 2)
    
    # Create convolution layer
    conv_layer = Conv2DGradFix(
        transpose=False,
        filters=weight.shape[-1],
        kernel_size=weight.shape[:2],
        strides=strides,
        padding=padding,
        dilation_rate=dilations,
        groups=groups,
        use_bias=bias is not None
    )
    
    # Forward pass
    def forward():
        return conv_layer(inputs)
    
    # Custom gradient function
    def grad(dy):
        # Gradient with respect to input
        if weight_gradients_disabled:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = forward()
            grad_inputs = tape.gradient(outputs, inputs, dy)
            grad_weight = None
            grad_bias = None
        else:
            with tf.GradientTape() as tape:
                tape.watch([inputs, weight])
                outputs = forward()
            grads = tape.gradient(outputs, [inputs, weight], dy)
            grad_inputs, grad_weight = grads
            
            # Compute bias gradients if needed
            if bias is not None:
                grad_bias = tf.reduce_sum(dy, axis=[0, 1, 2])
            else:
                grad_bias = None
        
        return grad_inputs, grad_weight, grad_bias
    
    # Perform forward pass and return with gradient function
    return forward(), grad

@tf.custom_gradient
def conv_transpose2d_gradfix(inputs, weight, bias=None, stride=1, padding='VALID',
                           output_padding=None, groups=1, dilation=1):
    """Custom transposed convolution operation with gradient fixes."""
    
    # Convert inputs to TensorFlow format
    strides = ensure_tuple(stride, 2)
    dilations = ensure_tuple(dilation, 2)
    
    # Create transposed convolution layer
    conv_layer = Conv2DGradFix(
        transpose=True,
        filters=weight.shape[-1],
        kernel_size=weight.shape[:2],
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilations,
        groups=groups,
        use_bias=bias is not None
    )
    
    # Forward pass
    def forward():
        return conv_layer(inputs)
    
    # Custom gradient function
    def grad(dy):
        # Gradient computation
        if weight_gradients_disabled:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                outputs = forward()
            grad_inputs = tape.gradient(outputs, inputs, dy)
            grad_weight = None
            grad_bias = None
        else:
            with tf.GradientTape() as tape:
                tape.watch([inputs, weight])
                outputs = forward()
            grads = tape.gradient(outputs, [inputs, weight], dy)
            grad_inputs, grad_weight = grads
            
            # Compute bias gradients if needed
            if bias is not None:
                grad_bias = tf.reduce_sum(dy, axis=[0, 1, 2])
            else:
                grad_bias = None
        
        return grad_inputs, grad_weight, grad_bias
    
    # Perform forward pass and return with gradient function
    return forward(), grad

def could_use_op(inputs):
    """Check if the custom convolution operation can be used."""
    if not enabled:
        return False
    
    # Check if input is on CPU
    return True

def conv2d(inputs, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    """Convolution operation with gradient fixes."""
    if could_use_op(inputs):
        return conv2d_gradfix(
            inputs=inputs,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
    
    # Fallback to regular convolution
    return tf.nn.conv2d(
        input=inputs,
        filters=weight,
        strides=stride,
        padding=padding,
        dilations=dilation
    )

def conv_transpose2d(inputs, weight, bias=None, stride=1, padding='VALID',
                    output_padding=None, groups=1, dilation=1):
    """Transposed convolution operation with gradient fixes."""
    if could_use_op(inputs):
        return conv_transpose2d_gradfix(
            inputs=inputs,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation
        )
    
    # Fallback to regular transposed convolution
    return tf.nn.conv2d_transpose(
        input=inputs,
        filters=weight,
        output_shape=None,  # Will be inferred
        strides=stride,
        padding=padding,
        dilations=dilation
    )

if __name__ == "__main__":
    # Test the implementation
    batch_size = 1
    height = 64
    width = 64
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    
    # Create test inputs
    inputs = tf.random.normal([batch_size, height, width, in_channels])
    weight = tf.random.normal([kernel_size, kernel_size, in_channels, out_channels])
    
    # Test regular convolution
    output = conv2d(inputs, weight)
    print(f"Conv2D output shape: {output.shape}")
    
    # Test transposed convolution
    weight_transpose = tf.random.normal([kernel_size, kernel_size, out_channels, in_channels])
    output_transpose = conv_transpose2d(inputs, weight_transpose)
    print(f"Conv2DTranspose output shape: {output_transpose.shape}")
