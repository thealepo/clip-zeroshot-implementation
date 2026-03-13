import torch
from torch import nn
from torch.nn import functional as F

class CLIP(nn.Module):
    def __init__(self , image_encoder , text_encoder , temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # Temperature that scales logits before softmax
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self , images , input_ids , attention_mask):
        # Get the raw image and text embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids , attention_mask)

        # L2 normalize the embeddings
        image_features = F.normalize(image_features , p=2 , dim=-1)
        text_features = F.normalize(text_features , p=2 , dim=-1)

        # Calculate the simmilarity matrix (logits)
        logits = (image_features @ text_features.T) / self.temperature

        return logits