import numpy as np
import tensorflow as tf


class Downsampler(tf.keras.layers.Layer):
    """
    Downsampler layer in TensorFlow.
    Implements the functionality of the PyTorch Downsampler class.
    """
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        
        assert phase in [0, 0.5], "phase should be 0 or 0.5"

        # Determine kernel properties based on type
        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1. / np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            raise ValueError("Wrong kernel type")

        # Generate kernel
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)

        # Create convolutional layer
        self.downsampler_ = tf.keras.layers.Conv2D(
            filters=n_planes,
            kernel_size=self.kernel.shape,
            strides=factor,
            padding="valid",
            use_bias=False,
            trainable=False
        )

        # Initialize convolutional layer weights with the kernel
        kernel_tensor = tf.constant(self.kernel, dtype=tf.float32)
        kernel_tensor = tf.expand_dims(kernel_tensor, axis=-1)  # Add input channels dimension
        kernel_tensor = tf.expand_dims(kernel_tensor, axis=-1)  # Add output channels dimension
        self.downsampler_.weights[0].assign(tf.tile(kernel_tensor, [1, 1, n_planes, n_planes]))

        # Handle padding if preserve_size is True
        self.preserve_size = preserve_size
        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                self.pad = (self.kernel.shape[0] - 1) // 2
            else:
                self.pad = (self.kernel.shape[0] - factor) // 2
            self.padding_layer = tf.keras.layers.ZeroPadding2D(padding=(self.pad, self.pad))

    def call(self, inputs):
        if self.preserve_size:
            inputs = self.padding_layer(inputs)
        return self.downsampler_(inputs)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    """
    Generate a resampling kernel for the downsampler.
    """
    assert kernel_type in ['lanczos', 'gauss', 'box']

    # Initialize kernel
    if phase == 0.5 and kernel_type != 'box':
        kernel = np.zeros([kernel_width - 1, kernel_width - 1], dtype=np.float32)
    else:
        kernel = np.zeros([kernel_width, kernel_width], dtype=np.float32)

    if kernel_type == 'box':
        assert phase == 0.5, "Box filter is always half-phased"
        kernel[:] = 1. / (kernel_width * kernel_width)

    elif kernel_type == 'gauss':
        assert sigma, "Sigma is not specified"
        assert phase != 0.5, "Phase 1/2 for gauss not implemented"

        center = (kernel_width + 1.) / 2.
        sigma_sq = sigma * sigma

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.
                dj = (j - center) / 2.
                kernel[i - 1, j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1, j - 1] /= (2. * np.pi * sigma_sq)

    elif kernel_type == 'lanczos':
        assert support, "Support is not specified"
        center = (kernel_width + 1.) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor

                val = 1
                if di != 0:
                    val *= support * np.sin(np.pi * di) * np.sin(np.pi * di / support) / (np.pi * np.pi * di * di)
                if dj != 0:
                    val *= support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support) / (np.pi * np.pi * dj * dj)

                kernel[i - 1, j - 1] = val

    else:
        raise ValueError("Wrong kernel type")

    kernel /= kernel.sum()
    return kernel


# Example usage
downsampler = Downsampler(n_planes=3, factor=2, kernel_type='lanczos2', phase=0.5, preserve_size=True)

# Test input
test_input = tf.random.normal([1, 256, 256, 3])  # Batch size 1, 256x256 image with 3 channels
output = downsampler(test_input)
print("Downsampled output shape:", output.shape)
