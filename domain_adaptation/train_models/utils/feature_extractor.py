import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


class FeatureExtractor(nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = resnet.fc.in_features
        # print(list(resnet.children()))

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)
