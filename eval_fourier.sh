#!/bin/bash

python src/eval_fourier.py  \
  --ckpt outputs/ckpt/vit_seed_1.pt \
  --batch_size 128 \
  --eps 4.0 \
  --max_batches 8