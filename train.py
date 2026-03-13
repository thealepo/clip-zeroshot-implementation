import os
import torch
from tqdm import tqdm
from transformers import DistilBertTokenizer
from src.dataset import get_dataloader
from src.models import CLIP
from src.loss import contrastive_loss
from src.utils import get_batch_embeddings

NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1
DATASET_ROOT = './MS-COCO-DATASET'
SAVE_PATH = 'clip_best_model.pth'

def train_epoch(model , dataloader , optimizer , device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader , desc='Training' , leave=False)

    for batch in progress_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        optimizer.zero_grad()
        logits = model(batch)
        loss = contrastive_loss(logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_epoch(model , dataloader , device):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader , desc='Validating' , leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            logits = model(batch)
            loss = contrastive_loss(logits)
            total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader , val_loader , val_dataset = get_dataloader(DATASET_ROOT , tokenizer , batch_size=128)

    model = CLIP().to(device)
    optimizer = torch.optim.AdamW(model.parameters() , lr=LEARNING_RATE , weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    fixed_val_batch = next(iter(val_loader))

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_epoch(model , train_loader , optimizer , device)
        val_loss = validate_epoch(model , val_loader , device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict() , SAVE_PATH)

if __name__ == '__main__':
    main()