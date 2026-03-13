import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from transformers import DistilBertModel
import numpy as np

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder , self).__init__()
        resnet_ = models.resnet50(weights='DEFAULT')
        self.image_encoder = nn.Sequential(*list(resnet_.children())[:-1])

    def forward(self , images):
        return torch.flatten(self.image_encoder(images) , 1)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder , self).__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    def forward(self , input_ids , attention_mask):
        outputs = self.model(input_ids , attention_mask)
        return outputs.last_hidden_state[:,0,:] # [CLS] token

class ProjectionHead(nn.Module):
    def __init__(self , embedding_dim , projection_dim):
        super(ProjectionHead , self).__init__()
        self.projection = nn.Linear(embedding_dim , projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim , projection_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self , x):
        x = self.projection(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

class CLIP(nn.Module):
    def __init__(self , projection_dim=512):
        super(CLIP , self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(768 , projection_dim)
        self.image_projection = ProjectionHead(2048 , projection_dim)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(self , batch):
        image_features = self.image_encoder(batch['pixel_values'])
        text_features = self.text_encoder(batch['input_ids'] , batch['attention_mask'])

        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        image_proj = F.normalize(image_proj , p=2 , dim=-1)
        text_proj = F.normalize(text_proj , p=2 , dim=-1)

        t = self.temp.exp()
        logits = (image_proj @ text_proj.T) / t
        return logits

