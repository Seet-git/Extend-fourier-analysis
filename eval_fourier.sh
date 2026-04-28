#!/bin/bash

python src/eval_fourier.py  \
  --ckpt_nat ./outputs/ckpt/vit_natural_seed_1.pt \
  --ckpt_gaussian ./outputs/ckpt/vit_gaussian_seed_1.pt \
  --ckpt_adv ./outputs/ckpt/vit_adversarial_seed_1.pt \
  --batch_size 128 \
  --eps 4.0