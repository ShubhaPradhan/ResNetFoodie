import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        return x