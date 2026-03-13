import torch
from torch import nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self , embedding_dim):
        super().__init__()
        # Load a pre-trained ResNet-50
        self.model = models.resnet50(pretrained=True)

        # Replace the last fully connected layer with a projection head
        self.model.fc = nn.Linear(self.model.fc.in_features , embedding_dim)

    def forward(self , x):
        return self.model(x)