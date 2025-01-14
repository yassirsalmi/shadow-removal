#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)
    
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im

def vis_parsing_maps_binary(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    part_colors = [[0,0,0]] + [[255, 255, 255]] * 15 + [[0, 0, 0]] * 19

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] + '-binary-color.jpg', vis_parsing_anno_color)

    return vis_parsing_anno_color[:,:,0]

def vis_parsing_maps_five_organs(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    brow = np.zeros(vis_parsing_anno.shape)
    eye = np.zeros(vis_parsing_anno.shape)
    nose = np.zeros(vis_parsing_anno.shape)
    mouse = np.zeros(vis_parsing_anno.shape)
    glass = np.zeros(vis_parsing_anno.shape)
    
    index_brow = np.where((vis_parsing_anno == 2) | (vis_parsing_anno == 3))
    index_eye = np.where((vis_parsing_anno == 4) | (vis_parsing_anno == 5))
    index_nose = np.where(vis_parsing_anno == 10)
    index_mouse = np.where((vis_parsing_anno == 11) | (vis_parsing_anno == 12) | (vis_parsing_anno == 13))
    index_glass = np.where(vis_parsing_anno == 6)

    brow[index_brow[0], index_brow[1]] = 1
    eye[index_eye[0], index_eye[1]] = 1
    nose[index_nose[0], index_nose[1]] = 1
    mouse[index_mouse[0], index_mouse[1]] = 1
    glass[index_glass[0], index_glass[1]] = 1

    return [brow, eye, nose, mouse, glass]

def vis_parsing_maps_binary_hair(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    part_colors = [[0,0,0]] + [[255, 255, 255]] * 16 + [[255,255,255]] + [[0, 0, 0]] * 17

    im = np.array(im)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
    
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path[:-4] + '-binary-color.jpg', vis_parsing_anno_color)

    return vis_parsing_anno_color[:,:,0]

def evaluate(respth='./res/test_res2', dspth='./data', cp='model_final_diss.pth', 
            seg_type='multiple', save_binary_mask=False, stride=1):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = tf.keras.models.load_model(cp)  # Load your TensorFlow model here
    
    normalize = tf.keras.layers.experimental.preprocessing.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    )
    
    def process_image(image_path):
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = tf.convert_to_tensor(np.array(image))
        img = tf.cast(img, tf.float32) / 255.0
        img = normalize(img)
        img = tf.expand_dims(img, 0)
        return img, image

    if os.path.isdir(dspth):
        for image_path in os.listdir(dspth):
            full_path = osp.join(dspth, image_path)
            img, image = process_image(full_path)
            
            out = net(img, training=False)
            parsing = tf.argmax(tf.squeeze(out[0] if isinstance(out, list) else out), axis=-1)
            parsing = parsing.numpy()
            
            if seg_type == 'multiple':
                mask = vis_parsing_maps(image, parsing, stride=1, 
                                     save_im=True, save_path=osp.join(respth, image_path))
            elif seg_type == 'binary':
                mask = vis_parsing_maps_binary(image, parsing, stride=1, 
                                            save_im=save_binary_mask, save_path=osp.join(respth, image_path))
            elif seg_type == 'five_organs':
                mask = vis_parsing_maps_five_organs(image, parsing, stride=1, 
                                                 save_im=save_binary_mask, save_path=osp.join(respth, image_path))
            elif seg_type == 'binary+hair':
                mask = vis_parsing_maps_binary_hair(image, parsing, stride=1, 
                                                 save_im=save_binary_mask, save_path=osp.join(respth, image_path))
    
    elif os.path.isfile(dspth):
        img, image = process_image(dspth)
        
        out = net(img, training=False)
        parsing = tf.argmax(tf.squeeze(out[0] if isinstance(out, list) else out), axis=-1)
        parsing = parsing.numpy()
        
        if seg_type == 'multiple':
            mask = vis_parsing_maps(image, parsing, stride=1, 
                                 save_im=True, save_path=osp.join(respth, os.path.basename(dspth)))
        elif seg_type == 'binary':
            mask = vis_parsing_maps_binary(image, parsing, stride=1, 
                                        save_im=save_binary_mask, save_path=osp.join(respth, os.path.basename(dspth)))
        elif seg_type == 'five_organs':
            mask = vis_parsing_maps_five_organs(image, parsing, stride=1, 
                                             save_im=save_binary_mask, save_path=osp.join(respth, os.path.basename(dspth)))
        elif seg_type == 'binary+hair':
            mask = vis_parsing_maps_binary_hair(image, parsing, stride=1, 
                                             save_im=save_binary_mask, save_path=osp.join(respth, os.path.basename(dspth)))
    
    return mask

if __name__ == "__main__":
    binary_mask = evaluate(
        respth='./res/test_res2',
        dspth='test-parsing',
        cp='face-seg-BiSeNet.h5',  # Update with your TensorFlow model path
        seg_type='binary'
    )
