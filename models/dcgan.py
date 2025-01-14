import tensorflow as tf


def dcgan(inp=2,
          ndf=32,
          num_ups=4, 
          need_sigmoid=True, 
          need_bias=True, 
          pad='zero', 
          upsample_mode='nearest', 
          need_convT=True):
    """
    Create a DCGAN model in TensorFlow.
    
    Args:
        inp (int): Number of input channels.
        ndf (int): Number of filters in the first layer.
        num_ups (int): Number of upsampling layers.
        need_sigmoid (bool): Whether to add a sigmoid activation at the end.
        need_bias (bool): Whether to include biases in layers.
        pad (str): Padding type ('zero' or other).
        upsample_mode (str): Upsampling mode ('nearest' or 'bilinear').
        need_convT (bool): Whether to use Conv2DTranspose or Upsampling + Conv2D.
    
    Returns:
        tf.keras.Sequential: A DCGAN generator model.
    """
    layers = [
        tf.keras.layers.Conv2DTranspose(ndf, kernel_size=3, strides=1, padding='valid', use_bias=need_bias),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.2),
    ]

    for _ in range(num_ups - 3):
        if need_convT:
            layers += [
                tf.keras.layers.Conv2DTranspose(ndf, kernel_size=4, strides=2, padding='same', use_bias=need_bias),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]
        else:
            layers += [
                tf.keras.layers.UpSampling2D(size=2, interpolation=upsample_mode),
                tf.keras.layers.Conv2D(ndf, kernel_size=3, strides=1, padding='same', use_bias=need_bias),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
            ]

    if need_convT:
        layers.append(
            tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=need_bias)
        )
    else:
        layers += [
            tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear'),
            tf.keras.layers.Conv2D(3, kernel_size=3, strides=1, padding='same', use_bias=need_bias),
        ]

    if need_sigmoid:
        layers.append(tf.keras.layers.Activation('sigmoid'))

    model = tf.keras.Sequential(layers)
    return model
