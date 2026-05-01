import argparse
import os
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T

# from models.ResNet import ResNet
# from models.ConvNeXt import ConvNeXt
from models.EfficientNet import EfficientNet
from torch.utils.data import DataLoader

from fourier.fourier_utils import save_heatmap, fourier_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_nat", type=str, required=True)
    parser.add_argument("--ckpt_gaussian", type=str, required=True)
    parser.add_argument("--ckpt_adv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=4.0)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    # Init env
    ckpts = {"natural": args.ckpt_nat, "gaussian": args.ckpt_gaussian, "adversarial": args.ckpt_adv}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    plot_path = Path("./outputs/plots")
    os.makedirs(plot_path, exist_ok=True)

    # Init loader
    test_transform = T.Compose([T.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Init model
    # model = ResNet(num_classes=10, dropout=0.1, pretrained=False).to(device)
    # model = ConvNeXt(num_classes=10, dropout=0.1, pretrained=False).to(device)
    model = EfficientNet(num_classes=10, dropout=0.1, pretrained=False).to(device)

    # Load model
    for mode, ckpt in ckpts.items():
        print(f"\t {mode} - Fourier heatmap ")
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Creat and save heatmap
        heatmap = fourier_heatmap(model=model, loader=test_loader, device=device, image_size=image_size, v_perturb=args.eps,
                                  ignore_edge_size=0, max_batches=args.max_batches)
        # save_heatmap(heatmap, f"{plot_path}/fourier_heatmap_resnet_{mode}.png")
        # save_heatmap(heatmap, f"{plot_path}/fourier_heatmap_convnext_{mode}.png")
        save_heatmap(heatmap, f"{plot_path}/fourier_heatmap_efficientnet_{mode}.png")


if __name__ == "__main__":
    main()