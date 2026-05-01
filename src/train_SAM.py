# general import
import os
import random
import argparse
from tqdm import tqdm
from pathlib import Path

# ML import
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as FT
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.losses import DiceLoss

# script import
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


class SAM2MaskDataset(Dataset):
    """
    images_dir/
        img_1.jpg
        img_2.jpg
    masks_dir/
        img_1.png
        img_2.png
    """

    def __init__(self, images_dir, masks_dir, is_train, args):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.is_train = is_train
        self.args = args
        self.suffix = [".png", ".jpg", ".jpeg"]
        image_paths = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in self.suffix])
        self.samples = []

        # Get all images/masks paths
        for image_path in image_paths:
            mask_path = None

            # Find mask
            for ext in self.suffix:
                is_mask_path = self.masks_dir / f"{image_path.stem}{ext}"
                if is_mask_path.exists():
                    mask_path = is_mask_path
                    break

            # Add image/mask
            if mask_path is not None:
                self.samples.append((image_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("Found 0 images in" + str(self.images_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # get image/mask
        image_path, mask_path = self.samples[idx]

        # load image rgb and mask grayscale
        input_img = Image.open(image_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Transform (paper-like)
        if self.is_train and random.random() < 0.5:
            input_img = FT.hflip(input_img)
            mask_img = FT.hflip(mask_img)

        # Apply augmentation on training set
        if self.is_train:
            input_img = apply_training_augmentation(input_img=input_img, args=self.args)

        # Convert mask to binary numpy array
        mask_np = np.asarray(mask_img).copy()
        mask_bin = (mask_np > 0).astype(np.float32)

        # Compute box
        mask_tensor = torch.as_tensor(mask_bin, dtype=torch.uint8)
        box = masks_to_boxes(mask_tensor[None, :, :]).squeeze(0)

        return {"image": np.asarray(input_img).copy(),
                "mask": mask_bin,
                "box": box}


def add_gaussian_noise(input_img, sigma):
    img = np.asarray(input_img).astype(np.float32) / 255.0
    sigma_e = np.random.rand() * sigma
    epsilon = sigma_e * np.random.randn(*img.shape).astype(np.float32)
    noisy_input = np.clip(img + epsilon, 0.0, 1.0)
    noisy_input = (noisy_input * 255.0).round().astype(np.uint8)
    return Image.fromarray(noisy_input, mode="RGB")


def apply_training_augmentation(input_img, args):
    if args.gaussian_augm:
        return add_gaussian_noise(input_img=input_img, sigma=args.sigma)

    if args.blur_augm:
        return T.GaussianBlur(kernel_size=5, sigma=(0.2, 2.0))(input_img)

    if args.color_augm:
        return T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))(input_img)

    if args.mixed_augm:
        modes = ["gaussian", "blur", "color"]
        random.shuffle(modes)
        for m in modes[: random.choice([1, 2])]:
            if m == "gaussian":
                input_img = add_gaussian_noise(input_img=input_img, sigma=args.sigma)
            elif m == "blur":
                input_img = T.GaussianBlur(kernel_size=5, sigma=(0.2, 2.0))(input_img)
            elif m == "color":
                input_img = T.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))(input_img)
        return input_img

    return input_img


def compute_batch_iou(pred_logits, target_masks) -> torch.Tensor:
    pred_masks = pred_logits > 0
    true_masks = target_masks > 0.5

    intersection = (pred_masks & true_masks).sum(dim=(1, 2, 3)).float()
    union = (pred_masks | true_masks).sum(dim=(1, 2, 3)).float()

    iou = intersection / union.clamp_min(1.0)
    return iou.mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Collate:
    def __init__(self, sam2_transforms, sam_output_size=(256, 256)):
        self.sam2_transforms = sam2_transforms
        self.sam_output_size = sam_output_size

    def __call__(self, batch):
        images_np = [b["image"] for b in batch]
        masks_np = [b["mask"] for b in batch]
        boxes_np = [b["box"] for b in batch]

        # Transform on CPU
        input_imgs = self.sam2_transforms.forward_batch(images_np)

        transformed_boxes = []
        target_masks = []

        for box_np, mask_np in zip(boxes_np, masks_np):
            # reshape for SAM2
            box = torch.as_tensor(box_np, dtype=torch.float32)
            box = self.sam2_transforms.transform_boxes(box[None, :], normalize=True, orig_hw=mask_np.shape)
            transformed_boxes.append(box.squeeze(0))

            # reshape to output SAM (256, 256)
            mask = torch.as_tensor(mask_np, dtype=torch.float32)[None, None, :, :]
            mask = F.interpolate(mask, size=self.sam_output_size, mode="nearest")
            target_masks.append(mask.squeeze(0))

        boxes = torch.stack(transformed_boxes, dim=0)
        masks = torch.stack(target_masks, dim=0)

        return {
            "images": input_imgs,
            "masks": masks,
            "boxes": boxes,
        }


def prepare_sam2_inputs(batch, device):
    input_imgs = batch["images"].to(device, non_blocking=True)
    target_masks = batch["masks"].to(device, non_blocking=True)
    boxes = batch["boxes"].to(device, non_blocking=True)

    return input_imgs, target_masks, boxes


def sam2_box_forward(model, input_imgs: torch.Tensor, boxes: torch.Tensor):
    batch_size = input_imgs.shape[0]

    # Source: sam2/sam2_image_predictor.py, SAM2ImagePredictor.__init__
    # Feature sizes
    bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor.set_image_batch
    # forward_image -> prepare_backbone_features
    backbone_out = model.forward_image(input_imgs)

    # Ref: sam2/modeling/sam2_base.py, SAM2Base._prepare_backbone_features
    # Flatten backbone FPN features to SAM2 internal layout
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor.set_image_batch
    # Add no memory embedding when the model config uses it
    if model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor.set_image_batch
    feats = [feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
             for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])][::-1]

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor.set_image_batch
    image_embed = feats[-1]
    high_res_feats = feats[:-1]

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    # Box format is reshaped to [B, 2, 2]
    if boxes.ndim == 2 and boxes.shape[-1] == 4:
        box_coords = boxes.reshape(batch_size, 2, 2)
    elif boxes.ndim == 3 and boxes.shape[-2:] == (2, 2):
        box_coords = boxes
    else:
        raise ValueError(f"Expected boxes shape [B, 4] or [B, 2, 2], got {boxes.shape}")

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    # 2 = top-left, 3 = bottom-right.
    box_labels = model._box_prompt_labels.expand(batch_size, -1)

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    concat_points = (box_coords, box_labels)

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    # Ref: sam2/modeling/sam2_base.py, SAM2Base._forward_sam_heads
    # Encode sparse and dense prompt embeddings.
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(points=concat_points, boxes=None, masks=None)

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    # Ref: sam2/modeling/sam2_base.py, SAM2Base._forward_sam_heads
    # Decode one mask per image because multimask_output=False.
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(image_embeddings=image_embed,
                                                                  image_pe=model.sam_prompt_encoder.get_dense_pe(),
                                                                  sparse_prompt_embeddings=sparse_embeddings,
                                                                  dense_prompt_embeddings=dense_embeddings,
                                                                  multimask_output=False,
                                                                  repeat_image=False,
                                                                  high_res_features=high_res_feats)

    # Ref: sam2/sam2_image_predictor.py, SAM2ImagePredictor._predict
    low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
    return low_res_masks, iou_predictions


def set_trainable_parameters(model):
    for p in model.parameters():
        p.requires_grad = True
    return


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_iou = torch.zeros((), device=device)
    total = 0

    for batch in tqdm(loader, desc="Evaluating"):
        # Prepare img, masks and prompts boxes
        input_imgs, target_masks, boxes = prepare_sam2_inputs(batch=batch, device=device)

        # Compute pred
        masks_pred, _ = sam2_box_forward(model=model, input_imgs=input_imgs, boxes=boxes)

        # Compute iou
        batch_size = input_imgs.shape[0]
        batch_iou = compute_batch_iou(
            pred_logits=masks_pred,
            target_masks=target_masks,
        )

        total_iou += batch_iou * batch_size
        total += batch_size

    return (total_iou / total).item()


def train(model, loader, optimizer, device, args):
    model.train()
    total_loss = torch.zeros((), device=device)
    total_iou = torch.zeros((), device=device)
    total = 0

    criterion_dice = DiceLoss(mode="binary", from_logits=True)

    # Loop into loader
    for batch in tqdm(loader, desc="Training"):
        # Prepare img, masks and prompts boxes
        input_imgs, target_masks, boxes = prepare_sam2_inputs(batch=batch, device=device)

        # Compute pred
        masks_pred, iou_predictions = sam2_box_forward(model=model, input_imgs=input_imgs, boxes=boxes)

        # Compute loss
        loss_bce = F.binary_cross_entropy_with_logits(masks_pred, target_masks)
        loss_dice = criterion_dice(masks_pred, target_masks)
        loss = args.bce_weight * loss_bce + args.dice_weight * loss_dice

        # Update
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = input_imgs.shape[0]
        total_loss += loss.detach() * batch_size

        # Compute iou
        with torch.no_grad():
            batch_iou = compute_batch_iou(pred_logits=masks_pred, target_masks=target_masks)
            total_iou += batch_iou * batch_size

        total += batch_size

    return (total_loss / total).item(), (total_iou / total).item()


def main():
    # naturally trained arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    # dataset arguments
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_masks", type=str, required=True)
    parser.add_argument("--val_images", type=str, required=True)
    parser.add_argument("--val_masks", type=str, required=True)

    # SAM2 repo
    parser.add_argument("--sam2_config", type=str, required=True,
                        help="e.g configs/sam2.1/sam2.1_hiera_t.yaml")

    # augmentation arguments
    training_mode = parser.add_mutually_exclusive_group()
    training_mode.add_argument("--gaussian_augm", action="store_true")
    training_mode.add_argument("--blur_augm", action="store_true")
    training_mode.add_argument("--color_augm", action="store_true")
    training_mode.add_argument("--mixed_augm", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.1, required=False)

    # finetuning arguments
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    args = parser.parse_args()

    # Init env
    set_seed(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(f"{args.output_dir}/ckpt")
    plot_path = Path(f"{args.output_dir}/plots")

    # Load SAM2 from config and checkpoint in training mode
    model = build_sam2(config_file=args.sam2_config, ckpt_path=None, device=device, mode="train",
                       apply_postprocessing=False)

    # Store box prompt once to avoid creating one at each forward
    model.register_buffer("_box_prompt_labels", torch.tensor([[2, 3]], dtype=torch.int, device=device),
                          persistent=False)
    # Apply basic transform SAM2
    sam2_transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0, max_hole_area=0.0,
                                     max_sprinkle_area=0.0)
    sam2_collate = Collate(sam2_transforms=sam2_transforms)

    # Freeze/unfreeze SAM2 TODO
    set_trainable_parameters(model=model)

    # Selected params for finetuning
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Prepare training set
    train_set = SAM2MaskDataset(images_dir=args.train_images, masks_dir=args.train_masks, is_train=True, args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, collate_fn=sam2_collate, persistent_workers=True)

    # Prepare test set
    test_set = SAM2MaskDataset(images_dir=args.val_images, masks_dir=args.val_masks, is_train=False, args=args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=sam2_collate, persistent_workers=True)

    mode = "natural"
    if args.gaussian_augm:
        mode = "gaussian"
    elif args.blur_augm:
        mode = "blur"
    elif args.color_augm:
        mode = "color"
    elif args.mixed_augm:
        mode = "mixed"

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    best_iou = 0.0
    val_iou = 0.0

    # Start training
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        loss, train_iou = train(model=model, loader=train_loader, optimizer=optimizer,
                                device=device, args=args)
        val_iou = evaluate(model=model, loader=test_loader, device=device)
        scheduler.step()
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f} | train_iou={train_iou:.4f} |"
                             f" val_iou={val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "sam2_config": args.sam2_config,
                        "mode": mode, "seed": args.seed, "epoch": epoch + 1, "val_iou": val_iou, "args": vars(args),
                        }, f"{model_path}/sam2_{mode}_seed_{args.seed}_best.pt")

    # Save last model
    torch.save({"model": model.state_dict(), "sam2_config": args.sam2_config,
                "mode": mode, "seed": args.seed, "epoch": args.epochs, "val_iou": val_iou, "args": vars(args),
                },
               f"{model_path}/sam2_{mode}_seed_{args.seed}_last.pt",
               )


if __name__ == "__main__":
    main()
