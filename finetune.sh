#!/bin/bash

SEEDS=(1) #  2 3 4 5 6 7 8 9 10
for SEED in "${SEEDS[@]}"
do
    echo "Training starts with seed $SEED"

    echo "==> Natural training"
    # python src/train_ResNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001
    # python src/train_ConvNeXt.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001
    python src/train_EfficientNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.001

    echo "==> Gaussian augmentation"
    # python src/train_ResNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001 --gaussian_augm
    # python src/train_ConvNeXt.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001 --gaussian_augm
    python src/train_EfficientNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.001 --gaussian_augm

    echo "==> Adversarial training"
    # python src/train_ResNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001 --adv_train --pgd_steps 2
    # python src/train_ConvNeXt.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.0001 --adv_train --pgd_steps 2
    python src/train_EfficientNet.py --seed $SEED --epochs 20 --batch_size 128 --lr 0.001 --adv_train --pgd_steps 2
done