import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXt(nn.Module):
    def __init__(self, num_classes, dropout, pretrained=True):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        pretrained_model = convnext_tiny(weights=weights)

        # Modify input layer for 32x32 inputs
        # Replace strided conv and layernorm with smaller stride 1 conv to avoid downsampling
        pretrained_model.features[0] = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([96, 32, 32])
        )

        # Feature extractor
        self.input_layer = pretrained_model.features[0]
        self.layer1 = pretrained_model.features[1]
        self.layer2 = pretrained_model.features[2]
        self.layer3 = pretrained_model.features[3]
        self.layer4 = pretrained_model.features[4]
        self.layer5 = pretrained_model.features[5]
        self.layer6 = pretrained_model.features[6]
        self.layer7 = pretrained_model.features[7]
        self.avgpool = pretrained_model.avgpool

        # Change classifier to match num_classes
        in_features = pretrained_model.classifier[2].in_features
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, input_img: Tensor):
        x = self.input_layer(input_img)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def main():
    model = ConvNeXt(num_classes=10,
                   dropout=0.1,
                   pretrained=False)

    # Random input
    x = torch.randn(4, 3, 32, 32)
    print("Input shape:", x.shape)
    out = model(x)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()