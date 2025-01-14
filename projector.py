import argparse
import math
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

import tensorflow as tf
# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

from model import Generator
from utils.projection_utils import *

def prepare_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for StyleGAN2 projection."
    )
    
    parser.add_argument(
        "--img_dir", type=str, default='imgs/', help="path to image dir to be projected"
    )
    parser.add_argument(
        "--files", type=str, default='', nargs="+", help="path to image files to be projected"
    )
    parser.add_argument(
        "--save_dir", type=str, default='results/', help="path to results to be saved"
    )
    parser.add_argument(
        "--ckpt", type=str, default='checkpoint/stylegan_tf', help="path to the StyleGAN model checkpoint"
    )
    parser.add_argument(
        "--face_seg_ckpt", type=str, default='checkpoint/face_seg_tf', help="path to the face segmentation model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument("--n_mean_latent", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--lr_G", type=float, default=0.1, help="learning rate of Generator")
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    
    return parser

def transform_image(img, size):
    img = tf.image.resize(img, [size, size])
    img = tf.image.resize_with_crop_or_pad(img, size, size)
    img = (img - 127.5) / 127.5
    return img

if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    n_mean_latent = 10000
    resize = min(args.size, 256)

    imgs = []
    if args.img_dir:
        args.files = [os.path.join(args.img_dir, fname) for fname in os.listdir(args.img_dir)]

    for imgfile in args.files:
        img = tf.io.read_file(imgfile)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = transform_image(img, resize)
        imgs.append(img)

    imgs = tf.stack(imgs, 0)

    g_ema = Generator(args.size, 512, 8)
    g_ema.trainable = False

    noise_sample = tf.random.normal([n_mean_latent, 512])
    latent_out = g_ema.style(noise_sample)
    latent_mean = tf.reduce_mean(latent_out, axis=0)
    latent_std = tf.sqrt(tf.reduce_sum(tf.pow(latent_out - latent_mean, 2)) / n_mean_latent)

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        shape = [imgs.shape[0]] + noise.shape[1:]
        noises.append(tf.Variable(tf.random.normal(shape)))

    latent_in = tf.Variable(tf.tile(tf.expand_dims(latent_mean, 0), [imgs.shape[0], 1]))

    if args.w_plus:
        latent_in = tf.Variable(tf.tile(tf.expand_dims(latent_in, 1), [1, g_ema.n_latent, 1]))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
    @tf.function
    def train_step(latent_in, noises, t):
        with tf.GradientTape() as tape:
            noise_strength = latent_std * args.noise * tf.maximum(0.0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength)

            img_gen = g_ema([latent_n], input_is_latent=True, noise=noises)

            if img_gen.shape[1] > 256:
                factor = img_gen.shape[1] // 256
                img_gen = tf.nn.avg_pool2d(img_gen, factor, factor, 'VALID')

            p_loss = tf.reduce_sum(tf.abs(img_gen - imgs))  
            n_loss = noise_regularize(noises)
            mse_loss = tf.reduce_mean(tf.square(img_gen - imgs))

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        gradients = tape.gradient(loss, [latent_in] + noises)
        optimizer.apply_gradients(zip(gradients, [latent_in] + noises))

        return loss, p_loss, n_loss, mse_loss

    latent_path = []
    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        tf.keras.backend.set_value(optimizer.lr, lr)

        loss, p_loss, n_loss, mse_loss = train_step(latent_in, noises, t)

        for noise in noises:
            noise_mean = tf.reduce_mean(noise)
            noise_std = tf.math.reduce_std(noise)
            noise.assign((noise - noise_mean) / noise_std)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.numpy())

        pbar.set_description(
            f"perceptual: {p_loss:.4f}; noise regularize: {n_loss:.4f};"
            f" mse: {mse_loss:.4f}; lr: {lr:.4f}"
        )

    final_latent = tf.convert_to_tensor(latent_path[-1])
    img_gen = g_ema([final_latent], input_is_latent=True, noise=noises)

    for i, input_name in enumerate(args.files):
        result_file = {
            "img": img_gen[i].numpy(),
            "latent": latent_in[i].numpy(),
            "noise": [n[i].numpy() for n in noises]
        }

        img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        img_array = ((img_gen[i].numpy() + 1) * 127.5).astype(np.uint8)
        Image.fromarray(img_array).save(os.path.join(args.save_dir, img_name))

        # Save latents and noise
        filename = os.path.splitext(os.path.basename(input_name))[0] + ".npz"
        np.savez(os.path.join(args.save_dir, filename), **result_file)
