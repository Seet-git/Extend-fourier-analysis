#!/bin/bash

python src/eval_fourier_ViT.py  \
  --ckpt_nat ./outputs/ckpt/vit_natural_seed_1.pt \
  --ckpt_gaussian ./outputs/ckpt/vit_gaussian_seed_1.pt \
  --ckpt_adv ./outputs/ckpt/vit_adversarial_seed_1.pt \
  --batch_size 128 \
  --eps 4.0

python src/eval_fourier_SAM.py \
  --ckpt_nat ./outputs/ecssd_fs/ckpt/sam2_natural_seed_1_best.pt \
  --ckpt_gaussian ./outputs/ecssd_fs/ckpt/sam2_gaussian_seed_1_best.pt \
  --ckpt_blur ./outputs/ecssd_fs/ckpt/sam2_blur_seed_1_best.pt \
  --ckpt_color ./outputs/ecssd_fs/ckpt/sam2_color_seed_1_best.pt \
  --ckpt_mixed ./outputs/ecssd_fs/ckpt/sam2_mixed_seed_1_best.pt \
  --val_images ./data/ecssd/test/images \
  --val_masks ./data/ecssd/test/masks \
  --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
  --batch_size 12 \
  --fourier_size 64 \
  --eps 64.0 \
  --size_thresholds 0.05 0.25 \
  --samples_per_size 20

python src/eval_fourier_SAM.py \
  --ckpt_nat ./outputs/coco/ckpt/sam2_natural_seed_1_best.pt \
  --ckpt_gaussian ./outputs/coco/ckpt/sam2_gaussian_seed_1_best.pt \
  --ckpt_blur ./outputs/coco/ckpt/sam2_blur_seed_1_best.pt \
  --ckpt_color ./outputs/coco/ckpt/sam2_color_seed_1_best.pt \
  --ckpt_mixed ./outputs/coco/ckpt/sam2_mixed_seed_1_best.pt \
  --val_images ./data/coco/test/images \
  --val_masks ./data/coco/test/masks \
  --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
  --batch_size 12 \
  --fourier_size 64 \
  --eps 64.0 \
  --size_thresholds 0.02 0.15 \
  --samples_per_size 20