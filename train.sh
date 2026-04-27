#!/bin/bash

SEEDS=(1) #  2 3 4 5 6 7 8 9 10
for SEED in "${SEEDS[@]}"
do
    echo "Training starts with seed $SEED"
    python src/train_ViT.py \
      --seed $SEED \
      --epochs 10 \
      --batch_size 128 \
      --lr 3e-4 \
      --gaussian_augm
done