import argparse
import os
import warnings
from pathlib import Path

# ML
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# import script
from train_SAM import SAM2MaskDataset, sam2_box_forward, compute_batch_iou
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from fourier.fourier_utils_SAM import (
    fourier_heatmap_sam,
    fourier_heatmaps_sam_by_size,
    save_heatmap,
)


class FourierCollate:
    def __init__(self, sam2_transforms, image_size, sam_output_size=(256, 256)):
        self.sam2_transforms = sam2_transforms
        self.image_size = image_size
        self.sam_output_size = sam_output_size

    def __call__(self, batch):
        images_01 = []
        target_masks = []
        transformed_boxes = []
        mask_area_abs = []
        mask_area_rel = []

        for sample in batch:
            image_np = sample["image"]
            mask_np = sample["mask"]
            box_np = sample["box"]

            # Resized to SAM2 input size
            image = torch.as_tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
            image = F.interpolate(image[None, :, :, :], size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False).squeeze(0)
            images_01.append(image)

            # Object size is computed on the gt mask before resizing
            mask_binary_np = mask_np > 0
            area_abs = float(mask_binary_np.sum())
            area_rel = area_abs / float(mask_binary_np.shape[0] * mask_binary_np.shape[1])
            mask_area_abs.append(torch.tensor(area_abs, dtype=torch.float32))
            mask_area_rel.append(torch.tensor(area_rel, dtype=torch.float32))

            # reshape for SAM2
            box = torch.as_tensor(box_np, dtype=torch.float32)
            box = self.sam2_transforms.transform_boxes(box[None, :], normalize=True, orig_hw=mask_np.shape)
            transformed_boxes.append(box.squeeze(0))

            # reshape to output SAM (256, 256)
            mask = torch.as_tensor(mask_np, dtype=torch.float32)[None, None, :, :]
            mask = F.interpolate(mask, size=self.sam_output_size, mode="nearest")
            target_masks.append(mask.squeeze(0))

        return {
            "images_01": torch.stack(images_01, dim=0),
            "masks": torch.stack(target_masks, dim=0),
            "boxes": torch.stack(transformed_boxes, dim=0),
            "mask_area_abs": torch.stack(mask_area_abs, dim=0),
            "mask_area_rel": torch.stack(mask_area_rel, dim=0),
        }


def _get_mask_ratio(sample):
    # get mask array
    mask_np = sample["mask"]
    mask_binary_np = mask_np > 0
    # count foreground pixels
    area_abs = float(mask_binary_np.sum())

    # return relative mask area (chatgpt)
    return area_abs / float(mask_binary_np.shape[0] * mask_binary_np.shape[1])


def _collect_mask_area(dataset, batch_size, max_batches):
    # choose number of samples
    if max_batches is None or batch_size is None:
        max_samples = len(dataset)
    else:
        max_samples = min(len(dataset), batch_size * max_batches)

    area_rels = []
    # compute relative area
    for idx in range(max_samples):
        area_rels.append(_get_mask_ratio(dataset[idx]))

    # return areas as tensor
    return torch.tensor(area_rels, dtype=torch.float32)


def compute_size_thresholds(area_relative, split_strategy, fix_thresholds):
    if split_strategy == "fixed":
        # use defined thresholds
        thresholds = tuple(float(x) for x in fix_thresholds)

    # split areas into 3 balanced groups
    elif split_strategy == "quantile":
        q = torch.quantile(area_relative, torch.tensor([1.0 / 3.0, 2.0 / 3.0], dtype=area_relative.dtype))
        thresholds = (float(q[0].item()), float(q[1].item()))

    return thresholds


def _get_size_group_index(area_relative, size_thresholds):
    small_thr, medium_thr = size_thresholds
    return {
        "small": torch.nonzero(area_relative < small_thr, as_tuple=False).flatten(),
        "medium": torch.nonzero((area_relative >= small_thr) & (area_relative < medium_thr), as_tuple=False).flatten(),
        "large": torch.nonzero(area_relative >= medium_thr, as_tuple=False).flatten(),
    }


def stratified_size_indices(area_rels, size_thresholds, samples_per_size, seed=0, shuffle_selected=True):
    """
    Chatgpt
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    group_to_candidates = _get_size_group_index(area_relative=area_rels, size_thresholds=size_thresholds)
    selected = []
    available_counts = {}
    selected_counts = {}

    for group in ("small", "medium", "large"):
        candidates = group_to_candidates[group]
        available_counts[group] = int(candidates.numel())
        if candidates.numel() == 0:
            selected_counts[group] = 0
            warnings.warn(f"No samples for size group '{group}'.")
            continue

        n_take = min(int(samples_per_size), int(candidates.numel()))
        perm = torch.randperm(candidates.numel(), generator=generator)[:n_take]
        chosen = candidates[perm]
        selected.extend(chosen.tolist())
        selected_counts[group] = int(n_take)

    if shuffle_selected and len(selected) > 1:
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        perm = torch.randperm(selected_tensor.numel(), generator=generator)
        selected = selected_tensor[perm].tolist()

    return selected, available_counts, selected_counts


def summarize_size_split(area_rels, size_thresholds):
    small_thr, medium_thr = size_thresholds

    counts = {
        "all": int(area_rels.numel()),
        "small": int((area_rels < small_thr).sum().item()),
        "medium": int(((area_rels >= small_thr) & (area_rels < medium_thr)).sum().item()),
        "large": int((area_rels >= medium_thr).sum().item()),
    }

    if area_rels.numel() > 0:
        stats = {
            "min_area_rel": float(area_rels.min().item()),
            "q33_area_rel": float(torch.quantile(area_rels, 1.0 / 3.0).item()),
            "median_area_rel": float(area_rels.median().item()),
            "q67_area_rel": float(torch.quantile(area_rels, 2.0 / 3.0).item()),
            "mean_area_rel": float(area_rels.mean().item()),
            "max_area_rel": float(area_rels.max().item()),
        }
    else:
        stats = {
            "min_area_rel": float("nan"),
            "q33_area_rel": float("nan"),
            "median_area_rel": float("nan"),
            "q67_area_rel": float("nan"),
            "mean_area_rel": float("nan"),
            "max_area_rel": float("nan"),
        }

    return counts, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_nat", type=str, default=None)
    parser.add_argument("--ckpt_gaussian", type=str, default=None)
    parser.add_argument("--ckpt_blur", type=str, default=None)
    parser.add_argument("--ckpt_color", type=str, default=None)
    parser.add_argument("--ckpt_mixed", type=str, default=None)

    parser.add_argument("--val_images", type=str, required=True)
    parser.add_argument("--val_masks", type=str, required=True)

    parser.add_argument("--sam2_config", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eps", type=float, default=4.0)
    parser.add_argument("--fourier_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sam_output_size", type=int, default=256)

    parser.add_argument("--no_size_analysis", action="store_true")
    parser.add_argument("--size_split", choices=("fixed", "quantile"), default="fixed")
    parser.add_argument("--size_thresholds", type=float, nargs=2, default=(0.01, 0.10),
                        metavar=("SMALL_THR", "MEDIUM_THR"))
    parser.add_argument("--samples_per_size", type=int, default=None)
    parser.add_argument("--size_sample_seed", type=int, default=0)
    parser.add_argument("--pred_threshold", type=float, default=0.0)
    parser.add_argument("--only_size_stats", action="store_true")
    args = parser.parse_args()

    # Init env
    ckpts = {
        "natural": args.ckpt_nat,
        "gaussian": args.ckpt_gaussian,
        "blur": args.ckpt_blur,
        "color": args.ckpt_color,
        "mixed": args.ckpt_mixed,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plot_path = Path("./outputs/plots")
    os.makedirs(plot_path, exist_ok=True)

    # Disable training augmentations during validation
    args.gaussian_augm = False
    args.blur_augm = False
    args.color_augm = False
    args.mixed_augm = False
    args.sigma = 0.1

    base_test_set = SAM2MaskDataset(images_dir=args.val_images, masks_dir=args.val_masks, is_train=False, args=args)
    eval_max_batches = args.max_batches

    if args.samples_per_size is not None:
        # Scan the full validation set so rare groups
        full_area_rels = _collect_mask_area(dataset=base_test_set, batch_size=None, max_batches=None)
        effective_size_thresholds = compute_size_thresholds(
            area_relative=full_area_rels,
            split_strategy=args.size_split,
            fix_thresholds=tuple(args.size_thresholds),
        )
        selected_indices, available_counts, selected_counts = stratified_size_indices(
            area_rels=full_area_rels,
            size_thresholds=effective_size_thresholds,
            samples_per_size=args.samples_per_size,
            seed=args.size_sample_seed,
            shuffle_selected=True,
        )

        test_set = Subset(base_test_set, selected_indices)
        eval_max_batches = None
    else:
        # Original behavior: inspect the exact first max_batches*batch_size samples.
        test_set = base_test_set
        area_rels = _collect_mask_area(dataset=test_set, batch_size=args.batch_size, max_batches=args.max_batches)
        effective_size_thresholds = compute_size_thresholds(
            area_relative=area_rels,
            split_strategy=args.size_split,
            fix_thresholds=tuple(args.size_thresholds),
        )

    if args.only_size_stats:
        return

    # Init model and SAM2 transforms
    model = build_sam2(config_file=args.sam2_config, ckpt_path=None, device=device, mode="eval",
                       apply_postprocessing=False)
    model.register_buffer("_box_prompt_labels", torch.tensor([[2, 3]], dtype=torch.int, device=device),
                          persistent=False)

    sam2_transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0, max_hole_area=0.0,
                                     max_sprinkle_area=0.0)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, collate_fn=FourierCollate(sam2_transforms=sam2_transforms,
                                                                        image_size=model.image_size,
                                                                        sam_output_size=(args.sam_output_size,
                                                                                         args.sam_output_size),
                                                                        ),
                             persistent_workers=args.num_workers > 0)

    for mode, ckpt in ckpts.items():
        print(f"\t {mode} - SAM2 Fourier heatmap ")
        state_dict = torch.load(ckpt, map_location=device)["model"]
        model.load_state_dict(state_dict)
        model.eval()

        if args.no_size_analysis:
            heatmap = fourier_heatmap_sam(model=model, loader=test_loader, device=device, image_size=model.image_size,
                                          fourier_size=args.fourier_size, v_perturb=args.eps,
                                          sam_forward_fn=sam2_box_forward, iou_fn=compute_batch_iou,
                                          ignore_edge_size=0, max_batches=eval_max_batches)
            save_heatmap(heatmap, f"{plot_path}/sam2_fourier_heatmap_{mode}.png",
                         title=f"SAM2 Fourier IoU degradation heatmap ({mode})",
                         label="IoU degradation", vmin=float(heatmap.min()),
                         vmax=float(heatmap.max()))
        else:
            heatmaps_by_size, counts_by_size = fourier_heatmaps_sam_by_size(
                model=model, loader=test_loader, device=device, image_size=model.image_size,
                fourier_size=args.fourier_size, v_perturb=args.eps, sam_forward_fn=sam2_box_forward,
                ignore_edge_size=0, max_batches=eval_max_batches, size_thresholds=effective_size_thresholds,
                pred_threshold=args.pred_threshold)

            print(f"Size counts {mode}: {counts_by_size}")

            for size_group, heatmap in heatmaps_by_size.items():
                valid = torch.isfinite(heatmap)
                save_heatmap(heatmap, f"{plot_path}/sam2_fourier_heatmap_{mode}_{size_group}.png",
                             title=f"SAM2 Fourier IoU degradation heatmap - {mode} ({size_group})",
                             label="IoU degradation",
                             vmin=float(heatmap[valid].min()),
                             vmax=float(heatmap[valid].max()))



if __name__ == "__main__":
    main()
