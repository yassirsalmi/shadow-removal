#!/bin/bash

CUDA_VISIBLE_DEVICES=""  

python3 remove_shadow.py \
    --save_dir results/ \
    --img_dir imgs/ \
    --size 256 \
    --ckpt checkpoint/stylegan_tf \
    --face_seg_ckpt checkpoint/face_seg_tf \
    --w_plus \
    --w_noise_reg 1e5 \
    --w_mse 1 \
    --w_percep 1 \
    --w_exclusion 0 \
    --w_arcface 0 \
    --fm_loss vgg \
    --detail_refine_loss \
    --visualize_detail \
    --step 1000 \
    --stage2 200 \
    --stage3 150 \
    --stage4 300 \
    --save_inter_res
