import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import time
import logging

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def make_tensor_to_np_image(tensor):
    """Convert TensorFlow tensor to numpy image array.
    Args:
        tensor: TensorFlow tensor of shape [B,H,W,C] in range [-1,1]
    Returns:
        Numpy array in range [0,255]
    """
    return tf.cast(
        tf.clip_by_value(
            (tensor + 1.0) * 127.5,
            0,
            255
        ),
        tf.uint8
    ).numpy()

def save_np_img(img_np, img_path):
    """Save numpy array as image."""
    img_np = img_np.astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(img_path)

def save_tf_img(img_tf, img_path):
    """Save TensorFlow tensor as image.
    Args:
        img_tf: tensor of shape [B,H,W,C] in range [-1,1]
        img_path: path to save the image
    """
    img_np = make_tensor_to_np_image(img_tf)
    
    num_imgs = img_np.shape[0]
    if num_imgs == 1:
        if img_np[0].shape[2] == 3:
            img_pil = Image.fromarray(img_np[0])
        elif img_np[0].shape[2] == 1:
            img_pil = Image.fromarray(img_np[0,:,:,0], 'L')
        img_pil.save(img_path)
    elif num_imgs > 1:
        for i in range(num_imgs):
            path, ext = osp.splitext(img_path)
            if img_np[0].shape[2] == 3:
                img_pil = Image.fromarray(img_np[i])
            elif img_np[0].shape[2] == 1:
                img_pil = Image.fromarray(img_np[i,:,:,0], 'L')
            img_pil.save(f"{path}_{i}{ext}")

def crop_image(img, d=32):
    """Make dimensions divisible by `d`."""
    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0])/2), 
        int((img.size[1] - new_size[1])/2),
        int((img.size[0] + new_size[0])/2),
        int((img.size[1] + new_size[1])/2),
    ]

    return img.crop(bbox)

def get_params(opt_over, model, net_input, downsampler=None):
    """Returns parameters that we want to optimize over."""
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
        if opt == 'net':
            params.extend(model.trainable_variables)
        elif opt == 'down':
            assert downsampler is not None
            params.extend(downsampler.trainable_variables)
        elif opt == 'input':
            params.append(net_input)
        else:
            raise ValueError(f'Unknown optimization parameter: {opt}')
            
    return params

def get_image_grid(images_np, nrow=8):
    """Creates a grid from a list of images."""
    # Convert to tensors
    images_tf = [tf.convert_to_tensor(x) for x in images_np]
    
    # Calculate grid dimensions
    n_images = len(images_tf)
    ncol = min(nrow, n_images)
    nrow = (n_images + ncol - 1) // ncol
    
    # Pad the list to fill the grid
    padding = ncol * nrow - n_images
    if padding > 0:
        images_tf.extend([tf.zeros_like(images_tf[0])] * padding)
    
    # Reshape into grid
    grid = tf.reshape(tf.stack(images_tf), [nrow, ncol, *images_tf[0].shape])
    
    # Transpose to get final image
    grid = tf.transpose(grid, [0, 2, 1, 3, 4])
    grid = tf.reshape(grid, [grid.shape[0] * grid.shape[1], 
                           grid.shape[2] * grid.shape[3], 
                           grid.shape[4]])
    
    return grid.numpy()

def plot_image_grid(images_np, nrow=8, factor=1, interpolation='lanczos'):
    """Draws images in a grid."""
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    # Ensure all images have same number of channels
    images_np = [x if (x.shape[0] == n_channels) else 
                np.concatenate([x, x, x], axis=0) for x in images_np]
    
    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[..., 0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid, interpolation=interpolation)
    
    plt.show()
    return grid

def load(path):
    """Load PIL image."""
    return Image.open(path)

def get_image(path, imsize=-1):
    """Load an image and resize to a specific size."""
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.LANCZOS)

    img_np = pil_to_np(img)
    return img, img_np

def fill_noise(shape, noise_type='u', var=1.0):
    """Generate noise tensor."""
    if noise_type == 'u':
        return tf.random.uniform(shape, -var, var)
    elif noise_type == 'n':
        return tf.random.normal(shape, 0, var)
    else:
        raise ValueError(f'Unknown noise type: {noise_type}')

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a tensor initialized in a specific way."""
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
        
    if method == 'noise':
        shape = [1, spatial_size[0], spatial_size[1], input_depth]
        net_input = fill_noise(shape, noise_type, var)
    elif method == 'meshgrid':
        assert input_depth == 2
        x = tf.linspace(0.0, 1.0, spatial_size[1])
        y = tf.linspace(0.0, 1.0, spatial_size[0])
        X, Y = tf.meshgrid(x, y)
        net_input = tf.stack([X, Y], axis=-1)
        net_input = tf.expand_dims(net_input, 0)
    else:
        raise ValueError(f'Unknown initialization method: {method}')
        
    return net_input

def pil_to_np(img_PIL):
    """Convert PIL image to numpy array."""
    ar = np.array(img_PIL)
    
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
        
    return ar.astype(np.float32) / 255.

def np_to_pil(img_np):
    """Convert numpy array to PIL image."""
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
        
    return Image.fromarray(ar)

def np_to_tf(img_np):
    """Convert numpy array to TensorFlow tensor."""
    # Convert from CHW to HWC format
    if len(img_np.shape) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    return tf.convert_to_tensor(img_np[None, ...])

def tf_to_np(img_tf):
    """Convert TensorFlow tensor to numpy array."""
    # Convert from NHWC to CHW format
    return np.transpose(img_tf.numpy()[0], (2, 0, 1))

def setup_logger(logpth):
    """Setup logging configuration."""
    logfile = f'BiSeNet-{time.strftime("%Y-%m-%d-%H-%M-%S")}.log'
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())
