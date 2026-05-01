#!/bin/bash


SEEDS=(1) #  2 3 4 5 6 7 8 9 10

# ViT
for SEED in "${SEEDS[@]}"
do
    echo "Training ViT starts with seed $SEED"

    COMMON_ARGS="\
      --seed $SEED \
      --epochs 200 \
      --batch_size 128 \
      --lr 1e-3"

    echo "==> Natural training"
    python src/train_ViT.py $COMMON_ARGS

    echo "==> Gaussian augmentation"
    python src/train_ViT.py $COMMON_ARGS --gaussian_augm

    echo "==> Adversarial training"
    python src/train_ViT.py $COMMON_ARGS --adv_train --pgd_steps 2
done

# SAM
for SEED in "${SEEDS[@]}"
do
    echo "Training SAM2 starts with seed $SEED"

    COMMON_ARGS="\
      --seed $SEED \
      --epochs 50 \
      --batch_size 24 \
      --lr 1e-5 \
      --train_images ./data/ecssd/trainval/images \
      --train_masks ./data/ecssd/trainval/masks \
      --val_images ./data/ecssd/test/images \
      --val_masks ./data/ecssd/test/masks \
      --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
      --output_dir outputs/ecssd"

    echo "==> SAM2 natural training"
    python src/train_SAM.py $COMMON_ARGS

    echo "==> SAM2 Gaussian augmentation"
    python src/train_SAM.py $COMMON_ARGS --gaussian_augm --sigma 0.1

    echo "==> SAM2 blur augmentation"
    python src/train_SAM.py $COMMON_ARGS --blur_augm

    echo "==> SAM2 color augmentation"
    python src/train_SAM.py $COMMON_ARGS --color_augm

    echo "==> SAM2 mixed augmentation"
    python src/train_SAM.py $COMMON_ARGS --mixed_augm
done

# SAM
for SEED in "${SEEDS[@]}"
do
    echo "Training SAM2 starts with seed $SEED"

    COMMON_ARGS="\
      --seed $SEED \
      --epochs 50 \
      --batch_size 24 \
      --lr 1e-5 \
      --train_images ./data/coco/trainval/images \
      --train_masks ./data/coco/trainval/masks \
      --val_images ./data/coco/test/images \
      --val_masks ./data/coco/test/masks \
      --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
      --output_dir outputs/coco"

    echo "==> SAM2 natural training"
    python src/train_SAM.py $COMMON_ARGS

    echo "==> SAM2 Gaussian augmentation"
    python src/train_SAM.py $COMMON_ARGS --gaussian_augm --sigma 0.1

    echo "==> SAM2 blur augmentation"
    python src/train_SAM.py $COMMON_ARGS --blur_augm

    echo "==> SAM2 color augmentation"
    python src/train_SAM.py $COMMON_ARGS --color_augm

    echo "==> SAM2 mixed augmentation"
    python src/train_SAM.py $COMMON_ARGS --mixed_augm
done

# SAM
for SEED in "${SEEDS[@]}"
do
    echo "Training SAM2 starts with seed $SEED"

    COMMON_ARGS="\
      --seed $SEED \
      --epochs 50 \
      --batch_size 24 \
      --lr 1e-5 \
      --train_images ./data/pets/trainval/images \
      --train_masks ./data/pets/trainval/masks \
      --val_images ./data/pets/test/images \
      --val_masks ./data/pets/test/masks \
      --sam2_config configs/sam2.1/sam2.1_hiera_t.yaml \
      --output_dir outputs/pets"

    echo "==> SAM2 natural training"
    python src/train_SAM.py $COMMON_ARGS

    echo "==> SAM2 Gaussian augmentation"
    python src/train_SAM.py $COMMON_ARGS --gaussian_augm --sigma 0.1

    echo "==> SAM2 blur augmentation"
    python src/train_SAM.py $COMMON_ARGS --blur_augm

    echo "==> SAM2 color augmentation"
    python src/train_SAM.py $COMMON_ARGS --color_augm

    echo "==> SAM2 mixed augmentation"
    python src/train_SAM.py $COMMON_ARGS --mixed_augm
done