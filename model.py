import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


# 1. Define the Multi-Output Model
class FaceAnalysisModel(nn.Module):
    def __init__(self, backbone_requires_grad: bool):
        super().__init__()
        # Load the pre-trained backbone
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.backbone = mobilenet_v3_large(weights=weights).features

        # 🧊 Freeze the backbone so we only train the heads
        for param in self.backbone.parameters():
            param.requires_grad = backbone_requires_grad

        # Global Pooling to shrink the spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Define the three heads
        # MobileNetV3-Large outputs 960 features before the classifier
        self.age_head = nn.Sequential(
            nn.Linear(960, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1) # Regression: single value
        )
        
        self.gender_head = nn.Sequential(
            nn.Linear(960, 64), 
            nn.ReLU(), 
            nn.Linear(64, 2) # Classification: Male or Female
        )

        self.ethnicity_head = nn.Sequential(
            nn.Linear(960, 128), 
            nn.ReLU(), 
            nn.Linear(128, 5) # Classification: 5 groups
        )

    def forward(self, x):
        # Repeat grayscale channel to 3 channels for MobileNet
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.age_head(x), self.gender_head(x), self.ethnicity_head(x)
