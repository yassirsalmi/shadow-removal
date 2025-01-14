import math
import tensorflow as tf

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def noise_regularize(noises):
    """Compute noise regularization loss.
    
    Args:
        noises: List of noise tensors
        
    Returns:
        Regularization loss tensor
    """
    loss = 0.0

    for noise in noises:
        size = tf.shape(noise)[2]

        while True:
            # Roll operations for spatial correlation loss
            rolled_right = tf.roll(noise, shift=1, axis=3)
            rolled_down = tf.roll(noise, shift=1, axis=2)
            
            # Compute mean correlations and square them
            horizontal_correlation = tf.reduce_mean(noise * rolled_right)
            vertical_correlation = tf.reduce_mean(noise * rolled_down)
            
            loss = loss + tf.pow(horizontal_correlation, 2) + tf.pow(vertical_correlation, 2)

            if size <= 8:
                break

            # Reshape and pool for multi-scale regularization
            noise = tf.reshape(noise, [-1, 1, size // 2, 2, size // 2, 2])
            noise = tf.reduce_mean(noise, axis=[3, 5])
            size = size // 2

    return loss

def noise_normalize_(noises):
    """Normalize noise tensors in-place.
    
    Args:
        noises: List of noise tensors
    """
    for i in range(len(noises)):
        mean = tf.reduce_mean(noises[i])
        std = tf.math.reduce_std(noises[i])
        
        # Update the tensor in-place
        noises[i] = (noises[i] - mean) / std

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    """Compute learning rate with cosine ramp-up and ramp-down.
    
    Args:
        t: Current time step
        initial_lr: Initial learning rate
        rampdown: Ramp-down period
        rampup: Ramp-up period
        
    Returns:
        Current learning rate
    """
    lr_ramp = tf.minimum(1.0, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * tf.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * tf.minimum(1.0, t / rampup)

    return initial_lr * lr_ramp

def latent_noise(latent, strength):
    """Add random noise to latent vectors.
    
    Args:
        latent: Input latent vectors
        strength: Noise strength
        
    Returns:
        Noisy latent vectors
    """
    noise = tf.random.normal(tf.shape(latent), dtype=latent.dtype) * strength
    return latent + noise

def make_image(tensor):
    """Convert normalized tensor to uint8 image.
    
    Args:
        tensor: Input tensor in range [-1, 1]
        
    Returns:
        uint8 numpy array in range [0, 255]
    """
    # Clamp values to [-1, 1]
    tensor = tf.clip_by_value(tensor, -1.0, 1.0)
    
    # Convert to [0, 255] range
    tensor = (tensor + 1) * 127.5
    
    # Convert to uint8
    tensor = tf.cast(tensor, tf.uint8)
    
    # Rearrange dimensions from NCHW to NHWC if needed
    if tensor.shape[1] in [1, 3, 4]:  # If in NCHW format
        tensor = tf.transpose(tensor, [0, 2, 3, 1])
    
    return tensor.numpy()

if __name__ == "__main__":
    # Test the functions
    
    # Test noise_regularize
    test_noises = [tf.random.normal([1, 1, 16, 16])]
    reg_loss = noise_regularize(test_noises)
    print("Regularization loss:", reg_loss.numpy())
    
    # Test noise_normalize_
    noise_normalize_(test_noises)
    print("Normalized noise mean:", tf.reduce_mean(test_noises[0]).numpy())
    print("Normalized noise std:", tf.math.reduce_std(test_noises[0]).numpy())
    
    # Test get_lr
    lr = get_lr(0.5, 0.1)
    print("Learning rate at t=0.5:", lr.numpy())
    
    # Test latent_noise
    latent = tf.zeros([1, 512])
    noisy_latent = latent_noise(latent, 0.1)
    print("Noisy latent std:", tf.math.reduce_std(noisy_latent).numpy())
    
    # Test make_image
    test_image = tf.random.uniform([1, 3, 64, 64], -1, 1)
    uint8_image = make_image(test_image)
    print("Output image shape:", uint8_image.shape)
    print("Output image dtype:", uint8_image.dtype)
    print("Output image range:", [uint8_image.min(), uint8_image.max()])
