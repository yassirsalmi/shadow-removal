import os
from PIL import Image
from tqdm import tqdm
import numpy as np

import tensorflow as tf
# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

import tensorflow_hub as tf_hub
import tensorflow_addons as tfa

from model import Generator, Discriminator
import utils
from utils import preprocess as preprocess
from projector import prepare_parser
from utils.projection_utils import *

def fill_noise(shape, noise_type):
    """Fills tensor with noise of type `noise_type`."""
    if noise_type == 'u':
        return tf.random.uniform(shape)
    elif noise_type == 'n':
        return tf.random.normal(shape)
    else:
        raise ValueError("Invalid noise type")

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way."""
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, spatial_size[0], spatial_size[1], input_depth]  # TF uses NHWC format
        net_input = fill_noise(shape, noise_type)
        net_input = net_input * var
    elif method == 'meshgrid':
        assert input_depth == 2
        x = tf.linspace(0.0, 1.0, spatial_size[1])
        y = tf.linspace(0.0, 1.0, spatial_size[0])
        X, Y = tf.meshgrid(x, y)
        meshgrid = tf.stack([X, Y], axis=0)
        net_input = tf.expand_dims(meshgrid, 0)
    else:
        raise ValueError("Invalid method")

    return net_input

def add_shadow_removal_parser(parser):
    parser.add_argument("--fm_loss", type=str, help="VGG or discriminator", choices=['disc', 'vgg'])
    parser.add_argument("--w_noise_reg", type=float, default=1e5)
    parser.add_argument("--w_mse", type=float, default=0)
    parser.add_argument("--w_percep", type=float, default=0)
    parser.add_argument("--w_arcface", type=float, default=0)
    parser.add_argument("--w_exclusion", type=float, default=0)
    parser.add_argument("--stage2", type=int, default=300)
    parser.add_argument("--stage3", type=int, default=450)
    parser.add_argument("--stage4", type=int, default=800)
    parser.add_argument("--detail_refine_loss", action='store_true')
    parser.add_argument("--visualize_detail", action='store_true')
    parser.add_argument("--save_samples", action='store_true')
    parser.add_argument("--save_inter_res", action='store_true')
    return parser

def preprocess_image(img_path, size):
    """Preprocess image using TensorFlow operations"""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [size, size])
    img = tf.image.resize_with_crop_or_pad(img, size, size)
    img = (img - 0.5) * 2.0  
    return img

class ShadowRemovalModel(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.output_h = 512
        self.output_w = 4
        
        self.g_ema = Generator(512, 512, 8)
        self.discriminator = Discriminator(args.size, channel_multiplier=2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        
        self.mask_net = self._create_mask_net()
        
        self._build_models()
        
        self.stage2 = tf.constant(args.stage2, dtype=tf.int32)
        self.stage3 = tf.constant(args.stage3, dtype=tf.int32)
        self.stage4 = tf.constant(args.stage4, dtype=tf.int32)
        self.w_noise_reg = tf.constant(args.w_noise_reg, dtype=tf.float32)
        self.w_mse = tf.constant(args.w_mse, dtype=tf.float32)
        self.w_percep = tf.constant(args.w_percep, dtype=tf.float32)
        
        self.shadow_matrix = self.add_weight(
            name="shadow_matrix",
            shape=(1, 1, 1, 3),
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        
        self.mask_noise = tf.Variable(
            tf.random.normal([1, self.output_h, self.output_w, 1]),
            trainable=True
        )

    def _build_models(self):
        dummy_style = tf.random.normal([1, 14, 512], dtype=tf.float32)
        dummy_latent = tf.random.normal([1, 14, 512], dtype=tf.float32)
        dummy_noise = tf.random.normal([1, 512, 4, 1], dtype=tf.float32)
        
        _ = self.g_ema([dummy_latent, dummy_style])
        
        dummy_img = tf.random.normal([1, 512, 4, 3], dtype=tf.float32)
        _ = self.discriminator(dummy_img)
        
        _ = self.mask_net(dummy_noise)

    def _create_mask_net(self):
        inputs = tf.keras.Input(shape=(512, 4, 1))
        x = tf.keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        outputs = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
        
        mask_net = tf.keras.Model(inputs=inputs, outputs=outputs, name='mask_network')
        
        # Manually build the model to create weights
        dummy_input = tf.zeros((1, 512, 4, 1), dtype=tf.float32)
        _ = mask_net(dummy_input)
        
        return mask_net

    def apply_shadow(self, img_gen):
        # Ensure mask_net is built and can be called
        if not hasattr(self.mask_net, 'built') or not self.mask_net.built:
            dummy_input = tf.zeros((1, 512, 4, 1), dtype=tf.float32)
            _ = self.mask_net(dummy_input)
        
        shadow_matrix = tf.sigmoid(self.shadow_matrix)
        shadow_reshaped = tf.reshape(shadow_matrix, [1, 1, 1, 3])
        img_gen_shadow = (img_gen + 1) * shadow_reshaped - 1
        
        # Ensure mask_noise has the correct shape
        if self.mask_noise.shape != (1, self.output_h, self.output_w, 1):
            self.mask_noise = tf.Variable(
                tf.random.normal([1, self.output_h, self.output_w, 1]),
                trainable=True
            )
        
        mask = self.mask_net(self.mask_noise)
        shadow_img = img_gen * mask + img_gen_shadow * (1 - mask)
        return shadow_img, mask

    def compute_loss(self, shadow_img, images, step):
        return tf.switch_case(
            tf.cast(
                tf.cast(step >= self.stage3, tf.int32) * 2 + 
                tf.cast(self.stage2 <= step < self.stage3, tf.int32), 
                tf.int32
            ),
            branch_fns={
                0: lambda: tf.reduce_mean(tf.abs(shadow_img - images)),
                1: lambda: tf.reduce_mean(tf.square(shadow_img - images)), 
                2: lambda: tf.reduce_mean(tf.abs(shadow_img - images))
            }
        )
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(1, 512, 4, 3), dtype=tf.float32, name='images'),
            tf.TensorSpec(shape=(1, 14, 512), dtype=tf.float32, name='latent_in'),
            tf.TensorSpec(shape=(1, 512, 4, 1), dtype=tf.float32, name='noises'),
            tf.TensorSpec(shape=(1, 512, 4, 3), dtype=tf.float32, name='binary_mask'),
            tf.TensorSpec(shape=(), dtype=tf.int32, name='step')
        ]
    )
    def train_step(self, images, latent_in, noises, binary_mask, step):
        trainable_vars = []
        trainable_vars.extend(self.g_ema.trainable_variables)
        trainable_vars.extend(self.discriminator.trainable_variables)
        trainable_vars.extend(self.mask_net.trainable_variables)
        trainable_vars.append(self.shadow_matrix)
        trainable_vars.append(self.mask_noise)

        with tf.GradientTape() as tape:
            style = tf.random.normal([1, 512], dtype=tf.float32)
            style = tf.expand_dims(style, 1)
            style = tf.broadcast_to(style, [1, 14, 512])
            
            img_gen = self.g_ema([latent_in, style], training=True)
            
            shadow_img, mask = self.apply_shadow(img_gen)
            
            loss = self.compute_loss(shadow_img, images, step)
                
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss, img_gen, shadow_img, mask

def load_stylegan_checkpoint(checkpoint_path):
    """Load the converted StyleGAN checkpoint."""
    try:
        # Load the SavedModel
        stylegan_model = tf.saved_model.load(checkpoint_path)
        return stylegan_model
    except Exception as e:
        print(f"Error loading StyleGAN checkpoint: {e}")
        return None

def load_face_segmentation_checkpoint(checkpoint_path):
    """Load the converted face segmentation checkpoint."""
    try:
        # Load the SavedModel
        face_seg_model = tf.saved_model.load(checkpoint_path)
        return face_seg_model
    except Exception as e:
        print(f"Error loading face segmentation checkpoint: {e}")
        return None

def main(img_path, res_dir, args):
    os.makedirs(res_dir, exist_ok=True)
    
    # Fixed sizes
    target_size = [512, 4]
    
    try:
        img = preprocess_image(img_path, 512)
        img = tf.cast(tf.expand_dims(img, 0), tf.float32)
        img = tf.image.resize(img, target_size)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return
    
    try:
        model = ShadowRemovalModel(args)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    stylegan_checkpoint_path = os.path.join('checkpoint', 'stylegan_tf')
    try:
        stylegan_model = load_stylegan_checkpoint(stylegan_checkpoint_path)
        if stylegan_model is None:
            raise ValueError("StyleGAN checkpoint could not be loaded")
        print("StyleGAN checkpoint loaded successfully")
    except Exception as e:
        print(f"Failed to load StyleGAN checkpoint: {e}")
        return
    
    face_seg_checkpoint_path = os.path.join('checkpoint', 'face_seg_tf')
    try:
        face_seg_model = load_face_segmentation_checkpoint(face_seg_checkpoint_path)
        if face_seg_model is None:
            raise ValueError("Face segmentation checkpoint could not be loaded")
        print("Face segmentation checkpoint loaded successfully")
    except Exception as e:
        print(f"Failed to load face segmentation checkpoint: {e}")
        return
    
    latent_in = tf.Variable(
        tf.random.normal([1, 14, 512], dtype=tf.float32),
        trainable=True
    )
    
    noises = tf.Variable(
        tf.random.normal([1, 512, 4, 1], dtype=tf.float32),
        trainable=True
    )
    
    binary_mask = tf.zeros([1, 512, 4, 3], dtype=tf.float32)
    
    final_loss = None
    final_img_gen = None
    final_shadow_img = None
    final_mask = None
    
    try:
        losses = []
        for step in tqdm(range(args.step)):
            try:
                loss, img_gen, shadow_img, mask = model.train_step(
                    img, latent_in, noises, binary_mask, step
                )
                
                final_loss = loss
                final_img_gen = img_gen
                final_shadow_img = shadow_img
                final_mask = mask
                
                losses.append(loss.numpy())
                
                if step % 50 == 0:
                    print(f"Step {step}: Loss = {loss.numpy()}")
                    
                    # Optional: Save intermediate results
                    if args.save_inter_res:
                        save_path = os.path.join(res_dir, f'step_{step}')
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Save generated images
                        tf.keras.utils.save_img(
                            os.path.join(save_path, 'gen.png'), 
                            tf.image.resize((img_gen[0] + 1) / 2, [256, 256])
                        )
                        tf.keras.utils.save_img(
                            os.path.join(save_path, 'shadow.png'), 
                            tf.image.resize((shadow_img[0] + 1) / 2, [256, 256])
                        )
                        tf.keras.utils.save_img(
                            os.path.join(save_path, 'mask.png'),
                            tf.image.resize(mask[0], [256, 256])
                        )
                        
            except Exception as step_error:
                print(f"Error in training step {step}: {step_error}")
                if not losses:
                    break
        
        if final_shadow_img is not None and losses:
            result_path = os.path.join(res_dir, 'shadow_removed.png')
            
            shadow_img_uint8 = tf.cast(
                tf.clip_by_value((final_shadow_img[0] + 1) * 127.5, 0, 255), 
                tf.uint8
            )
            
            result_img = Image.fromarray(shadow_img_uint8.numpy())
            result_img.save(result_path)
            print(f"Result saved to {result_path}")
        
        import json
        
        loss_data = {
            'losses': losses
        }
        
        with open(os.path.join(res_dir, 'loss_data.json'), 'w') as f:
            json.dump(loss_data, f)
        
        with open(os.path.join(res_dir, 'loss_curve.txt'), 'w') as f:
            f.write("Training Loss Curve\n")
            f.write("=" * 30 + "\n")
            
            f.write(f"Total steps: {len(losses)}\n")
            f.write(f"Initial loss: {losses[0]}\n")
            f.write(f"Final loss: {losses[-1]}\n")
            
            max_width = 50
            max_loss = max(losses)
            min_loss = min(losses)
            
            f.write("\nLoss Progression:\n")
            for i, loss in enumerate(losses):
                normalized_loss = (loss - min_loss) / (max_loss - min_loss) * max_width
                bar = '#' * int(normalized_loss)
                f.write(f"Step {i}: {loss:.4f} {'|' + bar}\n")
        
        print(f"Loss data saved to {os.path.join(res_dir, 'loss_data.json')}")
        print(f"Loss curve text visualization saved to {os.path.join(res_dir, 'loss_curve.txt')}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = prepare_parser()
    parser = add_shadow_removal_parser(parser)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    for img in os.listdir(args.img_dir):
        img_path = os.path.join(args.img_dir, img)
        img_name = os.path.splitext(img)[0]
        res_dir = os.path.join(args.save_dir, img_name)
        os.makedirs(res_dir, exist_ok=True)
        
        main(img_path, res_dir, args)
