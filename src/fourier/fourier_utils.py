# Code strongly inspired: https://github.com/gatheluck/FourierHeatmap

import torch
from tqdm import tqdm
import torch.fft as fft
import matplotlib.pyplot as plt
from typing import Iterator, Optional


def get_spectrum(height: int, width: int, ignore_edge_size: int = 0, low_center: bool = True, ) -> Iterator[
    torch.Tensor]:
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


def spectrum_to_basis(
        spectrum: torch.Tensor,
        image_size: int,
        device: torch.device,
        l2_normalize: bool = True,
) -> torch.Tensor:
    spectrum = spectrum.to(device)

    basis = fft.irfftn(
        spectrum,
        s=(image_size, image_size),
        dim=(-2, -1),
    )

    if l2_normalize:
        basis = basis / basis.norm()

    return basis


def create_fourier_heatmap_from_error_matrix(error_matrix: torch.Tensor) -> torch.Tensor:
    assert error_matrix.dim() == 2
    assert error_matrix.size(0) == 2 * (error_matrix.size(1) - 1)

    right_side = error_matrix[1:, :-1]
    left_side = torch.flip(right_side, dims=(0, 1))

    heatmap = torch.cat([left_side[:, :-1], right_side], dim=1)
    return heatmap


@torch.no_grad()
def fourier_heatmap(model, loader, device, image_size: int, v_perturb: float, ignore_edge_size: int = 0,
                    max_batches: Optional[int] = None) -> torch.Tensor:
    assert v_perturb > 0, "v_perturb must be > 0"

    model.eval()

    height = image_size
    width = image_size // 2 + 1

    fhmap_height = height - 2 * ignore_edge_size
    fhmap_width = width - ignore_edge_size

    error_matrix = torch.zeros(fhmap_height * fhmap_width, device=device, dtype=torch.float32)
    spectrums = get_spectrum(height=height, width=width, ignore_edge_size=ignore_edge_size, low_center=True)

    with tqdm(spectrums, total=fhmap_height * fhmap_width, ncols=120) as pbar:
        for idx, spectrum in enumerate(pbar):
            # U_ij, norme L2 = 1
            basis = spectrum_to_basis(
                spectrum=spectrum,
                image_size=image_size,
                device=device,
                l2_normalize=True,
            )
            basis = basis.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

            total = 0
            errors = 0

            for batch_idx, (input_img, label_img) in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                input_img = input_img.to(device)
                label_img = label_img.to(device)

                batch_size = input_img.size(0)
                channels = input_img.size(1)

                # Compute random sign
                sign = torch.randint(-1, 1, (batch_size, channels, 1, 1), device=device)
                sign[sign == 0] = 1
                sign = sign.float()

                # X_perturbed
                noise = sign * v_perturb * basis
                x_perturbed = input_img + noise
                x_perturbed = torch.clamp(x_perturbed, 0.0, 1.0)

                logits = model(x_perturbed)
                pred = logits.argmax(dim=1)

                errors += (pred != label_img).sum().item()
                total += label_img.size(0)

            error_rate = errors / total
            error_matrix[idx] = error_rate

            pbar.set_postfix({"err": error_rate})

    error_matrix = error_matrix.view(fhmap_height, fhmap_width)
    heatmap = create_fourier_heatmap_from_error_matrix(error_matrix)

    return heatmap.cpu()


def save_heatmap(heatmap: torch.Tensor, path: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Error rate")
    plt.title("Fourier Heat Map")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
