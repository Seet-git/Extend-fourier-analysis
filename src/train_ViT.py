# general import
import os
import random
import argparse
from tqdm import tqdm
from pathlib import Path

# ML import
import torch
import numpy as np
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

# script import
from models.ViT import ViT


def add_gaussian_noise(input_img, sigma):
    sigma_e = torch.rand(1, device=input_img.device) * sigma  # random [0, sigma]
    epsilon = sigma_e * torch.randn_like(input_img)
    noisy_input = input_img + epsilon
    noisy_input = torch.clamp(noisy_input, 0, 1)
    return noisy_input


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0.0
    total = 0.0

    for input_img, label_img in loader:
        input_img, label_img = input_img.to(device), label_img.to(device)

        # Compute pred
        logits = model(input_img)
        pred = logits.argmax(dim=1)

        # Compute output score
        correct += (pred == label_img).sum().item()
        total += label_img.size(0)

    return correct / total


def train(model, loader, optimizer, criterion, device, args):
    # Init training
    model.train()
    total_loss = 0.0
    total = 0

    # Loop into loader
    for input_img, label_img in loader:
        input_img, label_img = input_img.to(device), label_img.to(device)

        # Add gaussian noise at each training step
        if args.gaussian_augm:
            input_img = add_gaussian_noise(input_img=input_img, sigma=args.sigma)

        # Adversarial training PGD
        if args.adv_train:
            input_img = pgd_attack(model=model, input_img=input_img, label_img=label_img, criterion=criterion,
                                   epsilon=args.epsilon, alpha=args.alpha, steps=args.pgd_steps)

        # Compute loss
        optimizer.zero_grad()
        logits = model(input_img)
        loss = criterion(logits, label_img)

        # Update
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label_img.size(0)
        total += label_img.size(0)

    return total_loss / total


def pgd_attack(model, input_img, label_img, criterion, epsilon, alpha, steps):
    # Madry et al. cifar10_challenge (LinfPGDAttack)

    # input_img [0, 1] -> Madry param [0, 255] to [0, 1]
    eps = epsilon / 255
    step_size = alpha / 255

    x_nat = input_img.detach()

    # random start in L_inf ball
    x = x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)
    x = torch.clamp(x, 0, 1)

    for _ in range(steps):
        x.requires_grad_(True)

        loss = criterion(model(x), label_img)
        grad = torch.autograd.grad(loss, x)[0]

        # 1 pgd step
        x = x.detach() + step_size * grad.sign()
        x = torch.clamp(x, x_nat - eps, x_nat + eps)

        # valid image range [0, 1]
        x = torch.clamp(x, 0, 1).detach()

    return x


def main():
    # naturally trained arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    # gaussian + adv training arguments
    training_mode = parser.add_mutually_exclusive_group()
    training_mode.add_argument("--gaussian_augm", action="store_true")
    training_mode.add_argument("--adv_train", action="store_true")
    parser.add_argument("--sigma", type=float, default=0.1, required=False)
    parser.add_argument("--epsilon", type=float, default=8, required=False)
    parser.add_argument("--alpha", type=float, default=2, required=False)
    parser.add_argument("--pgd_steps", type=int, default=10, required=False)
    args = parser.parse_args()

    # Init env
    set_seed(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    padding = 4
    model_path = Path("./outputs/ckpt")
    plot_path = Path("./outputs/plots")

    # Transform (paper-like)
    train_transforms = [T.RandomCrop(image_size, padding=padding),
                        T.RandomHorizontalFlip(0.5),
                        T.ToTensor()]

    # Prepare transform
    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose([
        T.ToTensor(),
    ])

    # Prepare training est
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Prepare test set
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Prepare model, optimizer and loss
    model = ViT(input_size=image_size, patch_size=4, num_classes=10, dim=256, depth=6, heads=8, mlp_dim=512,
                dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    accuracy = 0.0

    # Start training
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        loss = train(model, train_loader, optimizer, criterion, device, args)
        accuracy = evaluate(model, test_loader, device)
        scheduler.step()
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f} | acc={accuracy:.4f}")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Save model
    mode = "natural"
    if args.gaussian_augm:
        mode = "gaussian"
    elif args.adv_train:
        mode = "adversarial"
    torch.save(model.state_dict(), f"{model_path}/vit_{mode}_seed_{args.seed}.pt")


if __name__ == "__main__":
    main()
