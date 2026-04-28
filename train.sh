#!/bin/bash

SEEDS=(1) #  2 3 4 5 6 7 8 9 10
for SEED in "${SEEDS[@]}"
do
    echo "Training starts with seed $SEED"

    echo "==> Natural training"
    python src/train_ViT.py --seed $SEED --epochs 100 --batch_size 128 --lr 0.001

    echo "==> Gaussian augmentation"
    python src/train_ViT.py --seed $SEED --epochs 100 --batch_size 128 --lr 0.001 --gaussian_augm

    echo "==> Adversarial training"
    python src/train_ViT.py --seed $SEED --epochs 100 --batch_size 128 --lr 0.001 --adv_train --pgd_steps 2
done