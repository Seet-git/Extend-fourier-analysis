# Code strongly inspired: https://github.com/gatheluck/FourierHeatmap

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.fft as fft
import matplotlib.pyplot as plt

SIZE_GROUPS = ("all", "small", "medium", "large")
RADIAL = (("very_low", 0.00, 0.10), ("low", 0.10, 0.25), ("mid", 0.25, 0.50),  # Chatgpt
          ("high", 0.50, 0.75), ("very_high", 0.75, 1.01))


def get_spectrum(height: int, width: int, ignore_edge_size: int = 0, low_center: bool = True):
    total = height * width
    indices = torch.arange(total)

    if low_center:
        indices = torch.cat([indices[total // 2:], indices[:total // 2]])

    indices = indices.view(height, width)

    if ignore_edge_size > 0:
        indices = indices[ignore_edge_size:-ignore_edge_size, :]
        indices = indices[:, :-ignore_edge_size]

    indices = indices.flatten()

    for idx in indices:
        spectrum = torch.nn.functional.one_hot(idx, num_classes=total)
        spectrum = spectrum.view(height, width).float()
        yield spectrum


def spectrum_to_basis(spectrum: torch.Tensor, fourier_size: int, image_size: int, device: torch.device,
                      l2_normalize: bool = True) -> torch.Tensor:
    spectrum = spectrum.to(device)
    basis = fft.irfftn(spectrum, s=(fourier_size, fourier_size), dim=(-2, -1))

    # Approximate coarse
    if fourier_size != image_size:
        basis = basis.unsqueeze(0).unsqueeze(0)
        basis = F.interpolate(basis, size=(image_size, image_size), mode="bilinear", align_corners=False)
        basis = basis.squeeze(0).squeeze(0)

    if l2_normalize:
        basis = basis / basis.norm()

    return basis


def fourier_heatmap_error_matrix(error_matrix: torch.Tensor) -> torch.Tensor:
    assert error_matrix.dim() == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    right_side = error_matrix[1:, :-1]
    left_side = torch.flip(right_side, dims=(0, 1))

    heatmap = torch.cat([left_side[:, :-1], right_side], dim=1)
    return heatmap


def normalize_sam2_input(images_01: torch.Tensor) -> torch.Tensor:
    # imagenet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=images_01.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images_01.device).view(1, 3, 1, 1)
    return (images_01 - mean) / std


def compute_per_sample_iou_from_logits(pred_logits: torch.Tensor, target_masks: torch.Tensor,
                                       threshold: float = 0.0, eps: float = 1e-6) -> torch.Tensor:
    if pred_logits.shape[-2:] != target_masks.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)

    pred = (pred_logits > threshold).float()
    target = (target_masks > 0.5).float()

    reduce_dims = tuple(range(1, pred.dim()))
    intersection = (pred * target).sum(dim=reduce_dims)
    union = (pred + target - pred * target).sum(dim=reduce_dims)

    # If both prediction and target are empty, IoU 1
    return torch.where(union > 0, intersection / (union + eps), torch.ones_like(union))


def size_group_masks(mask_area_rel: torch.Tensor,
                     size_thresholds):
    small_thr, medium_thr = size_thresholds
    return {"all": torch.ones_like(mask_area_rel, dtype=torch.bool),
            "small": mask_area_rel < small_thr,
            "medium": (mask_area_rel >= small_thr) & (mask_area_rel < medium_thr),
            "large": mask_area_rel >= medium_thr}


def _empty_error_accumulators(num_frequencies: int, device: torch.device):
    error_matrices = {group: torch.full((num_frequencies,), float("nan"), device=device, dtype=torch.float32)
                      for group in SIZE_GROUPS}
    count_matrices = {group: torch.zeros((num_frequencies,), device=device, dtype=torch.float32)
                      for group in SIZE_GROUPS}
    return error_matrices, count_matrices


def fourier_heatmap_sam(model, loader, device, image_size, fourier_size, v_perturb, sam_forward_fn,
                        iou_fn, ignore_edge_size=0, max_batches=None) -> torch.Tensor:
    with torch.no_grad():
        assert v_perturb > 0, "v_perturb must be > 0"
        model.eval()

        height = fourier_size
        width = fourier_size // 2 + 1

        fhmap_height = height - 2 * ignore_edge_size
        fhmap_width = width - ignore_edge_size

        error_matrix = torch.zeros(fhmap_height * fhmap_width, device=device, dtype=torch.float32)
        spectrums = get_spectrum(height=height, width=width, ignore_edge_size=ignore_edge_size, low_center=True)

        clean_total = 0
        clean_total_iou = torch.zeros((), device=device)
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_img = batch["images_01"].to(device, non_blocking=True)
            target_masks = batch["masks"].to(device, non_blocking=True)
            boxes = batch["boxes"].to(device, non_blocking=True)

            batch_size = input_img.size(0)
            input_img = normalize_sam2_input(input_img)

            masks_pred, _ = sam_forward_fn(model=model, input_imgs=input_img, boxes=boxes)
            batch_iou = iou_fn(pred_logits=masks_pred, target_masks=target_masks)

            clean_total_iou += batch_iou * batch_size
            clean_total += batch_size

        clean_iou = clean_total_iou / clean_total

        with tqdm(spectrums, total=fhmap_height * fhmap_width, ncols=120) as pbar:
            for idx, spectrum in enumerate(pbar):
                # U_ij, norme L2 = 1
                basis = spectrum_to_basis(spectrum=spectrum, fourier_size=fourier_size, image_size=image_size,
                                          device=device, l2_normalize=True)
                basis = basis.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                total = 0
                total_iou = torch.zeros((), device=device)

                for batch_idx, batch in enumerate(loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break

                    input_img = batch["images_01"].to(device, non_blocking=True)
                    target_masks = batch["masks"].to(device, non_blocking=True)
                    boxes = batch["boxes"].to(device, non_blocking=True)

                    batch_size = input_img.size(0)
                    channels = input_img.size(1)

                    # Compute random sign
                    sign = torch.randint(-1, 1, (batch_size, channels, 1, 1), device=device)
                    sign[sign == 0] = 1
                    sign = sign.float()

                    # X_perturbed
                    noise = sign * v_perturb * basis
                    x_perturbed = torch.clamp(input_img + noise, 0.0, 1.0)
                    input_img = normalize_sam2_input(x_perturbed)

                    masks_pred, _ = sam_forward_fn(model=model, input_imgs=input_img, boxes=boxes)

                    batch_iou = iou_fn(pred_logits=masks_pred, target_masks=target_masks)
                    total_iou += batch_iou * batch_size
                    total += batch_size

                mean_iou = total_iou / total
                delta_iou = clean_iou - mean_iou
                error_matrix[idx] = delta_iou

                pbar.set_postfix({"IoU": delta_iou.item()})

        error_matrix = error_matrix.view(fhmap_height, fhmap_width)
        heatmap = fourier_heatmap_error_matrix(error_matrix)

        return heatmap.cpu()


@torch.no_grad()
def fourier_heatmaps_sam_by_size(model, loader, device, image_size, fourier_size, v_perturb, sam_forward_fn,
                                 ignore_edge_size, max_batches, size_thresholds, pred_threshold):
    assert v_perturb > 0, "v_perturb must be > 0"
    model.eval()

    height = fourier_size
    width = fourier_size // 2 + 1

    fhmap_height = height - 2 * ignore_edge_size
    fhmap_width = width - ignore_edge_size
    num_frequencies = fhmap_height * fhmap_width

    #  Fourier basis and init error
    error_matrices, count_matrices = _empty_error_accumulators(num_frequencies=num_frequencies, device=device)
    spectrums = get_spectrum(height=height, width=width, ignore_edge_size=ignore_edge_size, low_center=True)

    # store clean IoU per size group
    clean_iou_sums = {group: torch.zeros((), device=device, dtype=torch.float32) for group in SIZE_GROUPS}
    clean_counts = {group: torch.zeros((), device=device, dtype=torch.float32) for group in SIZE_GROUPS}

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_img = batch["images_01"].to(device, non_blocking=True)
        target_masks = batch["masks"].to(device, non_blocking=True)
        boxes = batch["boxes"].to(device, non_blocking=True)
        mask_area_rel = batch["mask_area_rel"].to(device, non_blocking=True)

        input_img = normalize_sam2_input(input_img)
        masks_pred, _ = sam_forward_fn(model=model, input_imgs=input_img, boxes=boxes)

        # compute IoU per sample
        per_sample_iou = compute_per_sample_iou_from_logits(pred_logits=masks_pred, target_masks=target_masks,
                                                            threshold=pred_threshold)

        # split samples by object size
        group_masks = size_group_masks(mask_area_rel=mask_area_rel, size_thresholds=size_thresholds)
        for group, sample_mask in group_masks.items():
            group_count = sample_mask.sum().float()
            if group_count > 0:
                clean_iou_sums[group] += per_sample_iou[sample_mask].sum()
                clean_counts[group] += group_count

    #  mean IoU per group
    clean_mean_ious = {
        group: clean_iou_sums[group] / clean_counts[group]
        for group in SIZE_GROUPS
        if clean_counts[group] > 0
    }

    last_postfix = {}

    with tqdm(spectrums, total=num_frequencies, ncols=120) as pbar:
        for idx, spectrum in enumerate(pbar):
            # spectrum -> spatial perturbation
            basis = spectrum_to_basis(spectrum=spectrum, fourier_size=fourier_size, image_size=image_size,
                                      device=device, l2_normalize=True)
            basis = basis.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            # reset accumulators
            iou_sums = {group: torch.zeros((), device=device, dtype=torch.float32) for group in SIZE_GROUPS}
            counts = {group: torch.zeros((), device=device, dtype=torch.float32) for group in SIZE_GROUPS}

            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                input_img = batch["images_01"].to(device, non_blocking=True)
                target_masks = batch["masks"].to(device, non_blocking=True)
                boxes = batch["boxes"].to(device, non_blocking=True)
                mask_area_rel = batch["mask_area_rel"].to(device, non_blocking=True)

                batch_size = input_img.size(0)
                channels = input_img.size(1)

                # Compute random sign independently for sample/channel.
                sign = torch.randint(-1, 1, (batch_size, channels, 1, 1), device=device)
                sign[sign == 0] = 1
                sign = sign.float()

                # X_perturbed
                noise = sign * v_perturb * basis
                x_perturbed = torch.clamp(input_img + noise, 0.0, 1.0)
                input_img = normalize_sam2_input(x_perturbed)

                masks_pred, _ = sam_forward_fn(model=model, input_imgs=input_img, boxes=boxes)

                # compute IoU after perturbation
                per_sample_iou = compute_per_sample_iou_from_logits(
                    pred_logits=masks_pred,
                    target_masks=target_masks,
                    threshold=pred_threshold,
                )

                # group by object size
                group_masks = size_group_masks(mask_area_rel=mask_area_rel, size_thresholds=size_thresholds)
                for group, sample_mask in group_masks.items():
                    group_count = sample_mask.sum().float()
                    if group_count > 0:
                        iou_sums[group] += per_sample_iou[sample_mask].sum()
                        counts[group] += group_count

            for group in SIZE_GROUPS:
                if counts[group] > 0:
                    mean_iou = iou_sums[group] / counts[group]
                    error_matrices[group][idx] = clean_mean_ious[group] - mean_iou
                    count_matrices[group][idx] = counts[group]

            last_postfix = {group: None if counts[group] == 0 else round(float(error_matrices[group][idx].item()), 4)
                            for group in SIZE_GROUPS}
            pbar.set_postfix(last_postfix)

    heatmaps_by_size = {}
    counts_by_size = {}

    for group in SIZE_GROUPS:
        # reshape vector -> 2D Fourier map
        matrix = error_matrices[group].view(fhmap_height, fhmap_width)

        # convert to visual heatmap
        heatmaps_by_size[group] = fourier_heatmap_error_matrix(matrix).cpu()

        # number of samples used
        valid_counts = count_matrices[group][count_matrices[group] > 0]
        counts_by_size[group] = int(valid_counts[0].item()) if valid_counts.numel() > 0 else 0

    return heatmaps_by_size, counts_by_size


def save_heatmap(heatmap: torch.Tensor, path: str, title: str = "SAM2 Fourier heatmap",
                 label: str = "1 - IoU", vmin: float = 0.0, vmax: float = 1.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
