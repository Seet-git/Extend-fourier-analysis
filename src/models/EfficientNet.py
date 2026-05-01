import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNet(nn.Module):
    def __init__(self, num_classes, dropout, pretrained=True):
        super().__init__()

        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        pretrained_model = efficientnet_b4(weights=weights)

        # Modify input layer for 32x32 inputs
        # Replace strided conv with smaller stride 1 conv to avoid downsampling
        pretrained_model.features[0][0] = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1, bias=False)

        # Feature extractor
        self.input_layer = pretrained_model.features[0]
        self.layer1 = pretrained_model.features[1]
        self.layer2 = pretrained_model.features[2]
        self.layer3 = pretrained_model.features[3]
        self.layer4 = pretrained_model.features[4]
        self.layer5 = pretrained_model.features[5]
        self.layer6 = pretrained_model.features[6]
        self.layer7 = pretrained_model.features[7]
        self.layer8 = pretrained_model.features[8]
        self.avgpool = pretrained_model.avgpool

        # Change classifier to match num_classes
        in_features = pretrained_model.classifier[1].in_features
        self.classifier = nn.Sequential(
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
        x = self.layer8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    model = EfficientNet(num_classes=10,
                   dropout=0.1,
                   pretrained=False)

    # Random input
    x = torch.randn(4, 3, 32, 32)
    print("Input shape:", x.shape)
    out = model(x)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()