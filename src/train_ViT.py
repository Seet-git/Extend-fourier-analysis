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
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader

# script import
from models.ViT import ViT


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


def train(model, loader, optimizer, criterion, device):
    # Init training
    model.train()
    total_loss = 0.0
    total = 0

    # Loop into loader
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    return total_loss / total


def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--gaussian_augm", action="store_true", required=False)
    args = parser.parse_args()

    # Init env
    set_seed(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 32
    perturbation = 4
    model_path = Path("./outputs/ckpt")
    plot_path = Path("./outputs/plots")

    # Transform (paper-like)
    train_transforms = [T.Resize((image_size, image_size)),
                        T.RandomHorizontalFlip(),
                        T.ToTensor()]

    if args.gaussian_augm:
        pass
        # TODO

    # Prepare transform
    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose([
        T.Resize((image_size, image_size)),
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
    criterion = nn.CrossEntropyLoss()

    loss = 0.0
    accuracy = 0.0

    # Start training
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        loss = train(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, test_loader, device)
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | loss={loss:.4f} | acc={accuracy:.4f}")

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), f"{model_path}/vit_seed_{args.seed}.pt")


if __name__ == "__main__":
    main()
