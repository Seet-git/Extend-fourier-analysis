#!/bin/bash

python src/finetune_eval_fourier.py  \
  --ckpt_nat ./outputs/ckpt/efficientnet_natural_seed_1.pt \
  --ckpt_gaussian ./outputs/ckpt/efficientnet_gaussian_seed_1.pt \
  --ckpt_adv ./outputs/ckpt/efficientnet_adversarial_seed_1.pt \
  --batch_size 1024 \
  --eps 4.0

# --ckpt_nat ./outputs/ckpt/resnet_natural_seed_1.pt \
# --ckpt_gaussian ./outputs/ckpt/resnet_gaussian_seed_1.pt \
# --ckpt_adv ./outputs/ckpt/resnet_adversarial_seed_1.pt \
# --ckpt_nat ./outputs/ckpt/convnext_natural_seed_1.pt \
# --ckpt_gaussian ./outputs/ckpt/convnext_gaussian_seed_1.pt \
# --ckpt_adv ./outputs/ckpt/convnext_adversarial_seed_1.pt \