import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self, num_classes, dropout, pretrained=True):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        pretrained_model = resnet50(weights=weights)

        # Modify input layer for 32x32 inputs
        # Replace strided conv and maxpool with smaller stride 1 conv to avoid downsampling
        pretrained_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        pretrained_model.maxpool = nn.Identity()

        # Feature extractor
        self.input_layer = nn.Sequential(pretrained_model.conv1, pretrained_model.bn1, pretrained_model.relu)
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.avgpool = pretrained_model.avgpool

        # Change classifier to match num_classes
        in_features = pretrained_model.fc.in_features
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    model = ResNet(num_classes=10,
                   dropout=0.1,
                   pretrained=False)

    # Random input
    x = torch.randn(4, 3, 32, 32)
    print("Input shape:", x.shape)
    out = model(x)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()