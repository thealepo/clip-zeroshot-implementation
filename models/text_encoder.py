import torch
from torch import nn
from transformers import DistilBertModel

class TextEncoder(nn.Module):
    def __init__(self , embedding_dim):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Linear projection to the shared embedding space
        self.projection = nn.Linear(768 , embedding_dim)

    def forward(self , input_ids , attention_mask):
        output = self.model(input_ids , attention_mask)
        cls_embedding = output.last_hidden_state[:,0,:]
        return self.projection(cls_embedding)
