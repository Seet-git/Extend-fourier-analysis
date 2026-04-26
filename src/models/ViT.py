from typing import Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ViT(nn.Module):
    def __init__(self, input_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size

        # Image Patching and Embedding
        self.num_patches = int(math.pow(self.input_size // self.patch_size, 2))
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)  # Split + Flatten
        self.embedding = nn.Linear(patch_size * patch_size * 3, dim)

        # Positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        # Layer norm
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        # Attention mechanism
        self.heads = heads
        self.depth = depth
        self.multi_head = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Linear(dim, num_classes)

    def patch_input(self, img: Tensor):
        flatten = self.unfold(img).transpose(1, 2)
        emb = self.embedding(flatten)
        return emb

    def positional_encoding(self, x: Tensor):
        # Repeat for batch_size
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)

        # Add cls + pos
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        return x

    def transformer(self, in_emb: Tensor):
        for l in range(self.depth):
            x = self.layer_norm1(in_emb)
            x, _ = self.multi_head(x, x, x)
            att_layer = x + in_emb
            x = self.layer_norm2(att_layer)
            x = self.mlp(x)
            in_emb = x + att_layer
        return in_emb

    def forward(self, input_img: Tensor):
        x = self.patch_input(img=input_img)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        cls = x[:, 0, :]
        x = self.classifier(cls)
        return x


def main():
    model = ViT(input_size=224,
                patch_size=16,
                num_classes=10,
                dim=768,
                depth=12,
                heads=12,
                mlp_dim=3072,
                dropout=0.1)

    # Random input
    x = torch.randn(4, 3, 224, 224)
    print("Input shape:", x.shape)
    out = model(x)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
